import os
import re
import bz2
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

# 한국어 위키문헌 덤프에서 근현대시만 골라냄 (연 구분 빈 줄이 살아 있음)
dump_url = "https://dumps.wikimedia.org/kowikisource/latest/kowikisource-latest-pages-articles.xml.bz2"
dump_path = "./dataset/wikisource/kowikisource-latest-pages-articles.xml.bz2"
kpoem_path = "./dataset/kpoem/poems.parquet"
output_path = "./dataset/poems.parquet"  # KPoEM + 위키문헌을 합친 최종 시 데이터

# 1962년 이전에 사망해 저작권이 만료된 근현대 시인
poets = {
    "김소월", "한용운", "김억", "윤곤강", "정지용", "박용철", "김영랑", "심훈", "오장환",
    "최남선", "노천명", "임화", "이육사", "김기림", "오일도", "변영로", "이상", "윤동주",
    "김명순", "권구현", "이상화", "김동환", "이장희", "강경애", "홍사용", "조기천", "이광수",
    "정인보", "방정환", "권환", "남궁벽", "나혜석", "황석우", "김상용", "함형수", "박인환",
    "김대봉",
}
# 같은 시인이 쓴 것이라도 시가 아닌 장르는 제외
exclude_categories = {"시조", "한시", "향가", "가사", "고려속요", "수필", "단편소설", "장편소설", "악보", "번역"}
# 옛한글 자모, 또는 사용자 정의 영역에 넣은 옛 글자
old_hangul = re.compile(r"[ᄀ-ᇿㅤ-ㆎ-]")


def read_pages(path):
    for _, el in ET.iterparse(bz2.open(path)):
        if el.tag.endswith("}page"):
            ns = el.find("{*}ns").text
            title = el.find("{*}title").text
            text = el.find("{*}revision/{*}text").text or ""
            if ns == "0":
                yield title, text
            el.clear()


def get_field(text, names):
    m = re.search(r"\|\s*(?:" + "|".join(names) + r")\s*=\s*([^\n]*)", text)
    if not m:
        return None
    value = m.group(1)
    link = re.search(r"\[\[(?:저자|글쓴이):([^|\]]+)", value)
    return (link.group(1) if link else value).strip(" |}") or None


def clean(body):
    body = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", body, flags=re.S)
    body = re.sub(r"<sub>.*?</sub>", "", body)  # 한자 옆에 단 독음
    # {{윗주|黃昏|황혼}}, {{크게|...}} 같은 틀은 첫 번째 인자(원문)만 남김
    while re.search(r"\{\{[^{}]*\}\}", body):
        body = re.sub(r"\{\{[^{}|]*\|?([^{}|]*)[^{}]*\}\}", r"\1", body)
    body = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = body.replace("&nbsp;", " ").replace("'''", "").replace("''", "")
    # 目標[목표] 처럼 읽는 소리를 단 표기는 다른 시들처럼 목표(目標)로 통일
    body = re.sub(r"([一-鿿][一-鿿 ]*)\[([가-힣 ]+)\]", r"\2(\1)", body)
    body = re.sub(r"[①-⑳]", "", body)

    lines = [line.lstrip(":").strip() for line in body.split("\n")]
    lines = [line for line in lines if not line.startswith("註")]  # 각주
    body = "\n".join(lines)
    body = re.sub(r"\n〈[^〉]*〉\s*$", "", body.strip())  # 끝에 붙은 출처 표기 (예: 〈미발표(未發表)〉)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def shingles(text, n=4):
    text = re.sub(r"[^가-힣]", "", text)
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def build_poems(pages):
    poems = []
    for title, text in pages:
        blocks = re.findall(r"<poem[^>]*>(.*?)</poem>", text, flags=re.S)
        if not blocks or old_hangul.search(text):
            continue

        categories = {c.replace("‎", "").strip() for c in re.findall(r"\[\[분류:([^\]|]+)", text)}
        poet = get_field(text, ["저자", "지은이", "글쓴이", "author"])
        if poet not in poets or categories & exclude_categories:
            continue

        body = clean("\n\n".join(blocks))
        # 너무 짧거나 영어로 쓴 시는 제외
        if len(body) < 20 or len(re.findall(r"[A-Za-z]", body)) / len(body) > 0.3:
            continue

        poems.append({
            "poem_id": title,
            "title": clean(get_field(text, ["제목", "title"]) or title.split("/")[-1]),
            "poet": poet,
            "text": body,
        })

    return pd.DataFrame(poems)


def drop_duplicates(poems, kpoem):
    # 위키문헌 안에서 같은 시가 여러 페이지에 있는 경우
    poems = poems[~poems["text"].str.replace(r"\s", "", regex=True).duplicated()]

    # KPoEM에 이미 있는 시 (표기가 조금 달라도 4글자 조각이 30% 이상 겹치면 같은 시로 봄)
    kpoem_shingles = [(poet, shingles(text)) for poet, text in zip(kpoem["poet"], kpoem["text"])]
    keep = []
    for poet, text in zip(poems["poet"], poems["text"]):
        s = shingles(text)
        keep.append(not any(
            p == poet and len(s & k) / max(len(s | k), 1) >= 0.3
            for p, k in kpoem_shingles
        ))
    return poems[keep]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(dump_path):
        print(f"download: {dump_url}")
        urllib.request.urlretrieve(dump_url, dump_path)

    poems = build_poems(read_pages(dump_path))
    print(f"wikisource poems: {len(poems)}")

    kpoem = pd.read_parquet(kpoem_path)
    poems = drop_duplicates(poems, kpoem)
    print(f"after dedup: {len(poems)}")

    kpoem["source"] = "kpoem"
    poems["source"] = "wikisource"
    columns = ["source", "title", "poet", "text"]
    merged = pd.concat([kpoem[columns], poems[columns]], ignore_index=True)
    merged.to_parquet(output_path, index=False)

    print(f"total poems: {len(merged)}, chars: {merged['text'].str.len().sum()}")
    print(f"saved: {output_path}")
