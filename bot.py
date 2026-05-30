import os
import io
import base64
import json
import logging
from datetime import datetime
from collections import defaultdict
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import anthropic
import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_MAPS_API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 사용자별 세션
user_sessions = defaultdict(lambda: {
    "photos": [],       # {gps, timestamp, bytes} 리스트
    "trip_title": "",
    "memo": "",
    "waiting": False
})


# ─────────────────────────────────────────
# EXIF 유틸
# ─────────────────────────────────────────

def get_exif_data(image_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if not exif_data:
            return {}
        return {TAGS.get(tag_id, tag_id): value for tag_id, value in exif_data.items()}
    except Exception as e:
        logger.error(f"EXIF 추출 오류: {e}")
        return {}


def get_gps_coordinates(exif_data: dict) -> tuple | None:
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    try:
        def dms_to_decimal(dms, ref):
            d, m, s = dms
            dec = float(d) + float(m) / 60 + float(s) / 3600
            return -dec if ref in ["S", "W"] else dec

        lat = dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return (lat, lon)
    except Exception as e:
        logger.error(f"GPS 변환 오류: {e}")
        return None


def get_photo_timestamp(exif_data: dict) -> str | None:
    dt_str = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
    if dt_str:
        try:
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            return dt.strftime("%Y년 %m월 %d일 %H:%M")
        except:
            return dt_str
    return None


# ─────────────────────────────────────────
# ② Places API - 장소 컨텍스트 강화
# ─────────────────────────────────────────

async def gps_to_place_info(lat: float, lon: float) -> dict:
    """
    Google Places API (Nearby Search + Place Details)로
    장소명, 카테고리, 평점, 대표 리뷰까지 가져옴
    """
    async with httpx.AsyncClient() as client:

        # 1) Nearby Search로 가장 가까운 장소 찾기
        nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        nearby_params = {
            "location": f"{lat},{lon}",
            "radius": 100,          # 100m 이내
            "rankby": "prominence", # 유명도 순
            "key": GOOGLE_MAPS_API_KEY,
            "language": "ko"
        }
        nearby_resp = await client.get(nearby_url, params=nearby_params)
        nearby_data = nearby_resp.json()

        results = nearby_data.get("results", [])

        # 100m 내 결과 없으면 반경 확장
        if not results:
            nearby_params["radius"] = 500
            nearby_resp = await client.get(nearby_url, params=nearby_params)
            nearby_data = nearby_resp.json()
            results = nearby_data.get("results", [])

        if not results:
            # fallback: Geocoding API
            geo_url = "https://maps.googleapis.com/maps/api/geocode/json"
            geo_resp = await client.get(geo_url, params={
                "latlng": f"{lat},{lon}",
                "key": GOOGLE_MAPS_API_KEY,
                "language": "ko"
            })
            geo_data = geo_resp.json()
            address = ""
            if geo_data.get("results"):
                address = geo_data["results"][0].get("formatted_address", f"{lat:.4f}, {lon:.4f}")
            return {
                "name": address or f"{lat:.4f}, {lon:.4f}",
                "category": "장소",
                "rating": None,
                "reviews": [],
                "address": address
            }

        # 2) 가장 유명한 장소의 Place Details 가져오기
        place_id = results[0]["place_id"]
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "name,rating,reviews,types,formatted_address,editorial_summary",
            "key": GOOGLE_MAPS_API_KEY,
            "language": "ko"
        }
        details_resp = await client.get(details_url, params=details_params)
        details_data = details_resp.json()
        place = details_data.get("result", {})

        # 카테고리 한국어 매핑
        type_map = {
            "beach": "해변", "park": "공원", "museum": "박물관",
            "temple": "사원", "restaurant": "레스토랑", "cafe": "카페",
            "shopping_mall": "쇼핑몰", "tourist_attraction": "관광지",
            "lodging": "숙소", "art_gallery": "갤러리", "zoo": "동물원",
            "amusement_park": "놀이공원", "aquarium": "수족관",
            "night_club": "나이트클럽", "bar": "바", "spa": "스파",
            "natural_feature": "자연", "stadium": "경기장"
        }
        raw_types = place.get("types", [])
        category = next(
            (type_map[t] for t in raw_types if t in type_map),
            "장소"
        )

        # 리뷰 상위 2개 추출
        reviews = []
        for r in place.get("reviews", [])[:2]:
            text = r.get("text", "").strip()
            if text and len(text) > 10:
                reviews.append(text[:100])  # 100자 이내로 자름

        return {
            "name": place.get("name", results[0].get("name", "")),
            "category": category,
            "rating": place.get("rating"),
            "reviews": reviews,
            "address": place.get("formatted_address", ""),
            "editorial": place.get("editorial_summary", {}).get("overview", "")
        }


# ─────────────────────────────────────────
# ③ Claude Vision - 대표 사진 분석
# ─────────────────────────────────────────

def select_representative_photos(photos: list, max_count: int = 3) -> list:
    """
    사진 목록에서 대표 사진 max_count장 선택
    - GPS 있는 사진 우선
    - 시간순으로 첫/중간/마지막 분산 선택
    """
    gps_photos = [p for p in photos if p["gps"]]
    pool = gps_photos if gps_photos else photos

    if len(pool) <= max_count:
        return pool

    # 균등 분산 선택
    indices = [0, len(pool) // 2, len(pool) - 1]
    seen = set()
    selected = []
    for i in indices:
        if i not in seen:
            selected.append(pool[i])
            seen.add(i)
    return selected[:max_count]


def image_to_base64(image_bytes: bytes) -> str:
    """이미지를 base64로 인코딩 (Vision API용)"""
    # 용량 줄이기 위해 리사이즈
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


async def analyze_photos_with_vision(photos: list) -> list[str]:
    """
    Claude Vision으로 대표 사진들 분석
    각 사진에 대한 한국어 묘사 반환
    """
    rep_photos = select_representative_photos(photos, max_count=3)
    if not rep_photos:
        return []

    content = []
    for i, photo in enumerate(rep_photos, 1):
        b64 = image_to_base64(photo["bytes"])
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64
            }
        })
        content.append({
            "type": "text",
            "text": f"[사진 {i}]"
        })

    content.append({
        "type": "text",
        "text": (
            "위 사진들을 보고 각각 한 문장으로 묘사해주세요.\n"
            "형식: [사진1] 묘사 / [사진2] 묘사 / [사진3] 묘사\n"
            "묘사는 여행 블로그에 쓸 수 있는 감성적인 표현으로 작성해주세요.\n"
            "사람이 있으면 '아이', '딸', '가족' 등으로 표현하세요.\n"
            "JSON 없이 텍스트로만 반환하세요."
        )
    })

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    # "[사진1] ... / [사진2] ..." 형태 파싱
    descriptions = []
    for part in raw.split("/"):
        part = part.strip()
        # "[사진N]" 레이블 제거
        import re
        cleaned = re.sub(r"\[사진\d+\]\s*", "", part).strip()
        if cleaned:
            descriptions.append(cleaned)
    return descriptions


# ─────────────────────────────────────────
# 블로그 콘텐츠 생성
# ─────────────────────────────────────────

async def generate_blog_content(
    trip_title: str,
    memo: str,
    locations: list[dict],
    photo_descriptions: list[str]
) -> dict:
    """Claude API로 블로그 콘텐츠 생성"""

    # 장소 정보 텍스트 구성 (Places API 풍부한 데이터 포함)
    locations_text = ""
    for i, loc in enumerate(locations, 1):
        info = loc.get("place_info", {})
        name = info.get("name") or loc["place"]
        category = info.get("category", "")
        rating = info.get("rating")
        editorial = info.get("editorial", "")
        reviews = info.get("reviews", [])

        locations_text += f"{i}. {name}"
        if category:
            locations_text += f" [{category}]"
        if rating:
            locations_text += f" ⭐{rating}"
        if loc.get("time"):
            locations_text += f" ({loc['time']})"
        locations_text += "\n"
        if editorial:
            locations_text += f"   → {editorial}\n"
        for rv in reviews:
            locations_text += f"   💬 \"{rv}\"\n"

    # 사진 묘사 텍스트
    photos_text = ""
    if photo_descriptions:
        photos_text = "\n대표 사진 묘사 (글에 자연스럽게 녹여주세요):\n"
        for i, desc in enumerate(photo_descriptions, 1):
            photos_text += f"- {desc}\n"

    prompt = f"""당신은 감성적인 여행 블로그 작가입니다.

여행 제목: {trip_title}
추가 메모: {memo if memo else "없음"}
방문 장소 (시간순, 실제 장소 정보 포함):
{locations_text}
{photos_text}

아래 JSON 형식으로 정확히 반환하세요. JSON 외 다른 텍스트는 절대 포함하지 마세요.

{{
  "naver_blog": {{
    "intro": "도입부 (2-3문장, 감성적인 여행 시작 묘사)",
    "body": "본문 (각 장소별 3-4문장씩, 장소의 실제 분위기·특징·사진 묘사 자연스럽게 포함)",
    "outro": "마무리 (2-3문장, 하루를 돌아보는 감성 마무리)"
  }},
  "wordpress_english": {{
    "title": "SEO 친화적 영어 제목 (prokoreandad.com용)",
    "intro": "Opening paragraph (2-3 sentences, warm storytelling with Yuna)",
    "body": "Main content (3-4 sentences per place, include photo descriptions naturally)",
    "outro": "Closing paragraph (2-3 sentences, reflection on the day)"
  }},
  "instagram": {{
    "korean": "한국어 캡션 (감성적, 150자 이내, 해시태그 제외)",
    "english": "English caption (warm tone, 150 chars max, no hashtags)",
    "hashtags_korean": "#여행 #아이와여행 등 한국어 해시태그 10개",
    "hashtags_english": "#travel #familytravel etc 10 English hashtags"
  }}
}}"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    return json.loads(text)


def format_naver_html(naver: dict, trip_title: str) -> str:
    """네이버 블로그 HTML 에디터 호환 포맷"""
    intro = naver.get("intro", "")
    body = naver.get("body", "")
    outro = naver.get("outro", "")

    return (
        f"<h2>{trip_title}</h2>\n\n"
        f"<p>{intro}</p>\n\n"
        f"<p>{body}</p>\n\n"
        f"<p>{outro}</p>"
    )


def format_wordpress_text(wp: dict) -> str:
    """워드프레스 붙여넣기용 텍스트"""
    title = wp.get("title", "")
    intro = wp.get("intro", "")
    body = wp.get("body", "")
    outro = wp.get("outro", "")

    return (
        f"Title: {title}\n\n"
        f"{intro}\n\n"
        f"{body}\n\n"
        f"{outro}"
    )


# ─────────────────────────────────────────
# 텔레그램 핸들러
# ─────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✈️ *여행 블로그 봇에 오신 걸 환영해요!*\n\n"
        "사용법:\n"
        "1️⃣ `/new 발리 여행 1일차` — 새 여행 글 시작\n"
        "2️⃣ 사진을 *파일로* 전송 (GPS 보존됨) ← 중요!\n"
        "   📎 Telegram에서 첨부 → *파일로 보내기* 선택\n"
        "3️⃣ `/memo 오늘 유나가 처음 바다를 봤다` — 메모 추가 (선택)\n"
        "4️⃣ `/done` — 블로그 글 생성!\n\n"
        "⚠️ 일반 사진 전송은 Telegram이 GPS를 제거합니다.\n"
        "📍 Places API로 장소 정보 자동 수집\n"
        "📸 대표 사진 2~3장 Vision 분석 포함",
        parse_mode="Markdown"
    )


async def new_trip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    title = " ".join(context.args) if context.args else "여행"

    user_sessions[user_id] = {
        "photos": [],
        "trip_title": title,
        "memo": "",
        "waiting": True
    }

    await update.message.reply_text(
        f"📍 *{title}* 시작!\n\n"
        "사진을 전송해주세요. 여러 장 보내도 돼요.\n"
        "다 보내면 `/done` 입력!\n\n"
        "💡 `/memo [내용]` 으로 메모 추가 가능",
        parse_mode="Markdown"
    )


async def add_memo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    memo = " ".join(context.args)
    user_sessions[user_id]["memo"] = memo
    await update.message.reply_text(f"📝 메모 저장됨: {memo}")


async def _process_image(update: Update, image_bytes: bytes, session: dict):
    exif = get_exif_data(image_bytes)
    gps = get_gps_coordinates(exif)
    timestamp = get_photo_timestamp(exif)

    session["photos"].append({
        "gps": gps,
        "timestamp": timestamp,
        "bytes": image_bytes
    })

    count = len(session["photos"])
    gps_status = "📍 GPS 인식됨" if gps else "⚠️ GPS 없음"
    await update.message.reply_text(
        f"사진 {count}장 수신 {gps_status}\n"
        "계속 보내거나 `/done` 으로 완료!",
        parse_mode="Markdown"
    )


async def receive_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions[user_id]

    if not session["waiting"]:
        await update.message.reply_text(
            "먼저 `/new 여행제목` 으로 새 여행을 시작해주세요!",
            parse_mode="Markdown"
        )
        return

    # 일반 사진 전송은 Telegram이 EXIF를 제거함 — GPS 없음 가능성 높음
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = bytes(await file.download_as_bytearray())
    await _process_image(update, image_bytes, session)


async def receive_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """파일로 전송된 이미지 처리 — EXIF/GPS 원본 보존"""
    user_id = update.effective_user.id
    session = user_sessions[user_id]

    if not session["waiting"]:
        await update.message.reply_text(
            "먼저 `/new 여행제목` 으로 새 여행을 시작해주세요!",
            parse_mode="Markdown"
        )
        return

    doc = update.message.document
    file = await context.bot.get_file(doc.file_id)
    image_bytes = bytes(await file.download_as_bytearray())
    await _process_image(update, image_bytes, session)


async def done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions[user_id]

    if not session["photos"]:
        await update.message.reply_text("사진이 없어요! 먼저 사진을 보내주세요.")
        return

    await update.message.reply_text(
        f"⏳ {len(session['photos'])}장 분석 중...\n"
        "• 장소 정보 수집 (Places API)\n"
        "• 대표 사진 Vision 분석\n"
        "• 블로그 글 생성\n\n"
        "약 30~40초 소요됩니다!"
    )

    # ② Places API로 장소 정보 수집
    locations = []
    for photo in session["photos"]:
        if photo["gps"]:
            lat, lon = photo["gps"]
            place_info = await gps_to_place_info(lat, lon)
            place_name = place_info["name"]
        else:
            place_info = {}
            place_name = "위치 미확인"

        # 중복 장소 제거
        if not locations or locations[-1]["place"] != place_name:
            locations.append({
                "place": place_name,
                "time": photo["timestamp"],
                "place_info": place_info
            })

    # ③ Vision으로 대표 사진 분석
    photo_descriptions = []
    try:
        photo_descriptions = await analyze_photos_with_vision(session["photos"])
    except Exception as e:
        logger.error(f"Vision 분석 오류: {e}")
        # Vision 실패해도 계속 진행

    # 콘텐츠 생성
    try:
        content = await generate_blog_content(
            session["trip_title"],
            session["memo"],
            locations,
            photo_descriptions
        )

        naver = content.get("naver_blog", {})
        wp = content.get("wordpress_english", {})
        insta = content.get("instagram", {})

        # 장소 요약
        place_summary = "\n".join([f"• {l['place']}" for l in locations])
        vision_summary = ""
        if photo_descriptions:
            vision_summary = "\n\n📸 *사진 분석:*\n" + "\n".join([f"• {d}" for d in photo_descriptions])

        await update.message.reply_text(
            f"✅ *{session['trip_title']}* 완성!\n\n"
            f"📍 *인식된 장소:*\n{place_summary}"
            f"{vision_summary}",
            parse_mode="Markdown"
        )

        # 네이버 블로그 HTML
        naver_html = format_naver_html(naver, session["trip_title"])
        await update.message.reply_text(
            f"📝 *네이버 블로그 (HTML 에디터용)*\n\n{naver_html}",
            parse_mode="Markdown"
        )

        # 워드프레스
        wp_text = format_wordpress_text(wp)
        await update.message.reply_text(
            f"🌐 *WordPress (prokoreandad.com)*\n\n{wp_text}",
            parse_mode="Markdown"
        )

        # 인스타그램 (필드 분리)
        insta_kr = insta.get("korean", "")
        insta_en = insta.get("english", "")
        tags_kr = insta.get("hashtags_korean", "")
        tags_en = insta.get("hashtags_english", "")

        await update.message.reply_text(
            f"📸 *인스타그램*\n\n"
            f"🇰🇷 *한국어 캡션:*\n{insta_kr}\n\n"
            f"🇺🇸 *English caption:*\n{insta_en}\n\n"
            f"*한국어 해시태그:*\n{tags_kr}\n\n"
            f"*English hashtags:*\n{tags_en}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"콘텐츠 생성 오류: {e}")
        await update.message.reply_text(f"❌ 오류 발생: {str(e)}\n\n다시 시도해주세요.")

    # 세션 초기화
    user_sessions[user_id] = {
        "photos": [], "trip_title": "", "memo": "", "waiting": False
    }


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_trip))
    app.add_handler(CommandHandler("memo", add_memo))
    app.add_handler(CommandHandler("done", done))
    app.add_handler(MessageHandler(filters.PHOTO, receive_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, receive_document))

    logger.info("🚀 Travel Blog Bot 시작!")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
