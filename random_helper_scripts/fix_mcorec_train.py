import argparse

from lhotse import load_manifest, CutSet

def find_cut(cuts, sess_id):
    return [(i, cut) for i, cut in enumerate(cuts) if sess_id in cut.id][0]

def remove_speaker(cut: CutSet, speaker_id: str) -> CutSet:
    cut.supervisions = [sup for sup in cut.supervisions if sup.speaker != speaker_id]
    for key in cut.custom:
        if speaker_id in cut.custom[key]:
            del cut.custom[key][speaker_id]
    return cut

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fix mcorec training cutsets by removing invalid cuts (with no features)."
    )
    parser.add_argument("input_cutset", type=str, help="Path to the input CutSet manifest (JSON).")
    parser.add_argument("output_cutset", type=str, help="Path to the output fixed CutSet manifest (JSON).")
    args = parser.parse_args()

    print(f"Loading CutSet from {args.input_cutset}...")
    cuts = load_manifest(args.input_cutset)
    
    # Remove session 87
    cuts = cuts.filter(lambda c: '87' not in c.id)

    remove_speaker(find_cut(cuts, '26')[1], 'spk_2')
    remove_speaker(find_cut(cuts, '27')[1], 'spk_0')
    remove_speaker(find_cut(cuts, '28')[1], 'spk_2')
    remove_speaker(find_cut(cuts, '29')[1], 'spk_2')
    remove_speaker(find_cut(cuts, '30')[1], 'spk_2')

    cuts.to_file(args.output_cutset)
