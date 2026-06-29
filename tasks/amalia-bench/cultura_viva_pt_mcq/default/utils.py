import datasets


def process_docs_audiovisual(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include audiovisual domain."""
    return dataset.filter(lambda x: x["domain"] == "audiovisual")


def process_docs_festivals(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include festivals domain."""
    return dataset.filter(lambda x: x["domain"] == "festivals")


def process_docs_gastronomy(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include gastronomy domain."""
    return dataset.filter(lambda x: x["domain"] == "gastronomy")


def process_docs_geography(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include geography domain."""
    return dataset.filter(lambda x: x["domain"] == "geography")


def process_docs_holidays(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include holidays domain."""
    return dataset.filter(lambda x: x["domain"] == "holidays")


def process_docs_literature(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include literature domain."""
    return dataset.filter(lambda x: x["domain"] == "literature")


def process_docs_proverbs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include proverbs domain."""
    return dataset.filter(lambda x: x["domain"] == "proverbs")


def process_docs_heritage(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include heritage domain."""
    return dataset.filter(lambda x: x["domain"] == "heritage")


def process_docs_personalities(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include personalities domain."""
    return dataset.filter(lambda x: x["domain"] == "personalities")


def process_docs_sports(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter dataset to only include sports domain."""
    return dataset.filter(lambda x: x["domain"] == "sports")
