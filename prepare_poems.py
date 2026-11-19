import os
import pandas as pd

# KPoEM: 김소월, 윤동주, 임화, 이상, 한용운의 근현대시 483편 (행 단위)
# https://huggingface.co/datasets/AKS-DHLAB/KPoEM
source_url = "https://huggingface.co/datasets/AKS-DHLAB/KPoEM/resolve/main/KPoEM_line_dataset_v4.tsv"
output_path = "./dataset/kpoem/poems.parquet"


def build_poems(df):
    poems = []
    for poem_id, lines_df in df.groupby("poem_id", sort=True):
        lines = []
        prev_sub_title = None

        for sub_title, text in zip(lines_df["sub_title"], lines_df["text"]):
            # 소제목이 바뀌면 빈 줄을 두고 소제목을 넣음 (예: 오감도 시제15호의 일, 이, 삼)
            if pd.notna(sub_title) and sub_title != prev_sub_title:
                if lines:
                    lines.append("")
                lines.append(sub_title.strip())
                prev_sub_title = sub_title

            lines.append(text.strip())

        poems.append({
            "poem_id": poem_id,
            "title": lines_df["title"].iloc[0],
            "poet": lines_df["poet"].iloc[0],
            "text": "\n".join(lines),
        })

    return pd.DataFrame(poems)


if __name__ == "__main__":
    df = pd.read_csv(source_url, sep="\t").sort_values("line_id")
    poems = build_poems(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    poems.to_parquet(output_path, index=False)

    print(f"poems: {len(poems)}, chars: {poems['text'].str.len().sum()}")
    print(f"saved: {output_path}")
