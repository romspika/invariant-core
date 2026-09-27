# invariant-core/data/loaders/xnli_loader.py

from datasets import load_dataset
import pandas as pd
from collections import defaultdict

# Наши 8 сред: регион → языки XNLI
REGION_LANGUAGES = {
    "eastern_europe": ["ru", "bg"],        # болгарский есть в XNLI
    "western_europe": ["de", "fr", "es"],
    "anglosphere":    ["en"],
    "arab_world":     ["ar"],
    "south_asia":     ["hi", "ur"],
    "east_asia":      ["zh"],
    "southeast_asia": ["th", "sw"],        # суахили как прокси Африки
    "latin_america":  ["es"],              # дублирует, но с другим split
}

# XNLI содержит 15 языков: ar, bg, de, el, en, es, fr, hi,
# ru, sw, th, tr, ur, vi, zh

def load_xnli_by_region(split="validation", samples_per_lang=5000):
    """
    Загружает XNLI, группирует по регионам-средам.
    split: "train" | "validation" | "test"
    samples_per_lang: сколько примеров брать на язык
    """

    # XNLI загружается отдельно на каждый язык
    region_data = defaultdict(list)
    lang_stats = {}

    for region, languages in REGION_LANGUAGES.items():
        print(f"\n→ Загрузка региона: {region}")

        for lang in languages:
            try:
                # Загружаем датасет для языка
                dataset = load_dataset(
                    "xnli",
                    lang,
                    split=split,
                    trust_remote_code=True
                )

                # Берём нужное количество
                n = min(samples_per_lang, len(dataset))
                subset = dataset.select(range(n))

                # Добавляем метаданные
                for item in subset:
                    region_data[region].append({
                        "premise":    item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label":      item["label"],  # 0=entailment, 1=neutral, 2=contradiction
                        "language":   lang,
                        "region":     region,
                        # Позже добавим: source_type, time_period
                    })

                lang_stats[lang] = n
                print(f"  ✓ {lang}: {n} примеров")

            except Exception as e:
                print(f"  ✗ {lang}: ошибка — {e}")

    return region_data, lang_stats


def region_data_to_df(region_data):
    """Конвертирует в DataFrame для удобного анализа."""
    all_rows = []
    for region, items in region_data.items():
        all_rows.extend(items)
    return pd.DataFrame(all_rows)


def print_stats(df):
    """Базовая статистика по средам."""
    print("\n=== СТАТИСТИКА ПО СРЕДАМ ===")
    print(df.groupby("region")["label"].value_counts().unstack(fill_value=0))
    print(f"\nВсего примеров: {len(df)}")
    print(f"Языков: {df['language'].nunique()}")
    print(f"Регионов: {df['region'].nunique()}")

    # Баланс — важно для IRM
    sizes = df.groupby("region").size()
    print(f"\nРазмеры сред (мин/макс): {sizes.min()} / {sizes.max()}")
    ratio = sizes.max() / sizes.min()
    if ratio > 3:
        print(f"⚠ Дисбаланс {ratio:.1f}x — нужна стратификация")
    else:
        print(f"✓ Баланс в норме ({ratio:.1f}x)")


if __name__ == "__main__":
    print("Загружаем XNLI validation split...")
    region_data, lang_stats = load_xnli_by_region(
        split="validation",
        samples_per_lang=5000
    )

    df = region_data_to_df(region_data)
    print_stats(df)

    # Сохраняем
    df.to_parquet("xnli_by_region.parquet", index=False)
    print("\n✓ Сохранено: xnli_by_region.parquet")
