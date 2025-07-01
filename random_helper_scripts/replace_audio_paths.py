import sys

from lhotse import load_manifest, CutSet
from tqdm import tqdm


def replace_audio_paths(cutset: CutSet, old_path: str, new_path: str):
    new_cuts = []
    for cut in tqdm(cutset):
        for src in cut.recording.sources:
            src.source = src.source.replace(old_path, new_path)
        new_cuts.append(cut)
    return CutSet.from_cuts(new_cuts)


if __name__ == "__main__":
    cutset = load_manifest(sys.argv[1])
    old_path = sys.argv[2]
    new_path = sys.argv[3]
    cutset = replace_audio_paths(cutset, old_path, new_path)
    cutset.to_file(sys.argv[1])
