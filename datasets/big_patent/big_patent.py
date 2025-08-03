import os
import datasets

_DESCRIPTION = """\
BIGPATENT is a dataset consisting of 1.3 million U.S. patent documents along with human-written abstractive summaries.
"""

_HOMEPAGE = "https://huggingface.co/datasets/NortheasternUniversity/big_patent"
_LICENSE = "MIT"

class BigPatentConfig(datasets.BuilderConfig):
    def __init__(self, name, **kwargs):
        super(BigPatentConfig, self).__init__(name=name, **kwargs)

class BigPatent(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BigPatentConfig(name=name, version=datasets.Version("1.0.0"), description=f"BIGPATENT {name} split")
        for name in list("abcdefghij")
    ]

    DEFAULT_CONFIG_NAME = "a"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                
    "document": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "application_number": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        local_data_dir = os.path.abspath("data/bigpatent_c")
        file_path = os.path.join(local_data_dir, f"{self.config.name}.json.gz")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Please download it manually.")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": file_path},
            )
        ]

    def _generate_examples(self, filepath):
        import gzip
        import json

        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                yield idx, {
                    "document": data.get("description", ""),
                    "abstract": data.get("abstract", ""),
                    "title": data.get("title", ""),
                    "application_number": data.get("application_number", ""),
                }
