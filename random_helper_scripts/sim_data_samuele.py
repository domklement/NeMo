import logging
import os.path
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lhotse
import numpy as np
import pyroomacoustics
import scipy.signal
import soundfile as sf
import torch
import torchaudio
from fontTools.merge.util import current_time
from lhotse import (
    AudioSource,
    CutSet,
    MonoCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    dill_enabled,
)
from lhotse.cut import CutSet, MixedCut, MixTrack
from lhotse.cut.set import mix
from lhotse.manipulation import combine as combine_manifests
from lhotse.parallel import parallel_map
from lhotse.recipes.librispeech import prepare_librispeech
from lhotse.supervision import AlignmentItem, SupervisionSegment
from lhotse.utils import add_durations, uuid4
from lhotse.workflows.meeting_simulation.base import (
    MAX_TASKS_WAITING,
    BaseMeetingSimulator,
    MeetingSampler,
    reverberate_cuts,
)
from scipy.signal import fftconvolve, firwin
from tqdm import tqdm

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# we can perturb the speed lazily of each utterance.
SPEED_PERTURB_LOW = 0.95
SPEED_PERTURB_HIGH = 1.05

# NOTE: we cannot use randomly sampled RIRs because the model may learn to distinguish speakers based on
# irrealistic RIR differences.
# rir sim parameters here
# each session will be in a single room.
# speaker position may change
RT60 = (0.1, 0.6)
ROOM_SZ = (5, 20)
ROOM_CEILING = (3, 6)
DELTA_DIST = 0.5  # min distance between walls and among speakers
RAN_DISP = 0.05


def split_monocut_at_pauses(
        monocut: MonoCut, pause_threshold: float = 0.2
) -> List[MonoCut]:
    """
    Split a MonoCut at pauses longer than the specified threshold while preserving alignments.

    When splitting at gaps:
    - The previous segment ends at gap_start + gap_duration/2
    - The next segment starts at gap_start + gap_duration/2
    - Leading silence longer than threshold creates a separate silent segment
    - Trailing silence longer than threshold creates a separate silent segment
    - For gaps shorter than threshold, segments include the full gap

    Args:
        monocut: The MonoCut to split
        pause_threshold: Minimum pause duration in seconds to split at (default: 0.2s = 200ms)

    Returns:
        List of MonoCut objects split at long pauses
    """

    if not monocut.supervisions or not monocut.supervisions[0].alignment:
        # No alignment info, return original cut
        return [monocut]

    supervision = monocut.supervisions[0]
    word_alignments = supervision.alignment["word"]

    # Filter out empty symbols upfront
    valid_alignments = [item for item in word_alignments if item.symbol.strip() != ""]

    if len(valid_alignments) == 0:
        return [monocut]  # No valid words, return original

    # Check for leading silence
    first_word = valid_alignments[0]
    leading_silence = first_word.start - monocut.start

    # Check for trailing silence
    last_word = valid_alignments[-1]
    last_word_end = last_word.start + last_word.duration
    monocut_end = monocut.start + monocut.duration
    trailing_silence = monocut_end - last_word_end

    # Find all potential split points (including leading/trailing silence)
    split_points = []
    segment_boundaries = []  # (start_word_idx, end_word_idx, segment_start_time, segment_end_time)

    # Handle leading silence
    current_segment_start_time = monocut.start
    current_word_start_idx = 0

    if leading_silence >= pause_threshold:
        # Leading silence gets split in the middle
        split_time = monocut.start + leading_silence / 2
        split_points.append(split_time)
        current_segment_start_time = split_time

    # Find gaps between words that exceed threshold
    for i in range(len(valid_alignments) - 1):
        current_item = valid_alignments[i]
        next_item = valid_alignments[i + 1]

        current_end = current_item.start + current_item.duration
        gap_start = current_end
        gap_end = next_item.start
        gap_duration = gap_end - gap_start

        if gap_duration >= pause_threshold:
            # End current segment at middle of gap
            split_time = gap_start + gap_duration / 2

            # Add current segment
            segment_boundaries.append((
                current_word_start_idx,
                i,  # end at current word (inclusive)
                current_segment_start_time,
                split_time
            ))

            # Start new segment at same split point
            current_segment_start_time = split_time
            current_word_start_idx = i + 1
            split_points.append(split_time)

    # Handle the final segment
    final_segment_end_time = monocut_end
    if trailing_silence >= pause_threshold:
        # Split trailing silence in the middle
        split_time = last_word_end + trailing_silence / 2
        # End the words segment at the split
        segment_boundaries.append((
            current_word_start_idx,
            len(valid_alignments) - 1,
            current_segment_start_time,
            split_time
        ))
        split_points.append(split_time)
    else:
        # Include trailing silence in the final segment
        segment_boundaries.append((
            current_word_start_idx,
            len(valid_alignments) - 1,
            current_segment_start_time,
            final_segment_end_time
        ))

    # If no splits were made, return original
    if not split_points:
        return [monocut]

    # Create new MonoCuts for each segment
    result_cuts = []
    for seg_idx, (start_word_idx, end_word_idx, segment_start, segment_end) in enumerate(segment_boundaries):
        segment_words = valid_alignments[start_word_idx:end_word_idx + 1]

        if not segment_words:
            continue

        segment_duration = segment_end - segment_start

        # Adjust alignment times to be relative to segment start
        adjusted_alignments = []
        for item in segment_words:
            adjusted_item = AlignmentItem(
                symbol=item.symbol,
                start=item.start - segment_start,
                duration=item.duration,
                score=item.score,
            )
            adjusted_alignments.append(adjusted_item)

        # Create segment text
        segment_text = " ".join(item.symbol for item in segment_words)

        # Create new supervision with adjusted alignments
        new_supervision = SupervisionSegment(
            id=f"{supervision.id}-{seg_idx:03d}",
            recording_id=supervision.recording_id,
            start=0,  # Relative to the new cut
            duration=segment_duration,
            channel=supervision.channel,
            text=segment_text,
            language=supervision.language,
            speaker=supervision.speaker,
            gender=supervision.gender,
            custom=supervision.custom,
            alignment={"word": adjusted_alignments},
        )

        # Create new MonoCut
        new_cut = MonoCut(
            id=f"{monocut.id}-{seg_idx:03d}",
            start=segment_start,
            duration=segment_duration,
            channel=monocut.channel,
            supervisions=[new_supervision],
            recording=monocut.recording,
            custom=monocut.custom,
        )
        result_cuts.append(new_cut)

    return result_cuts

def split_monocuts_batch(
    monocuts: List[MonoCut], pause_threshold: float = 0.2, num_jobs: int = 1
) -> List[MonoCut]:
    """
    Split a list of MonoCuts at pauses longer than the specified threshold.

    Args:
        monocuts: List of MonoCuts to split
        pause_threshold: Minimum pause duration in seconds to split at (default: 0.2s = 200ms)

    Returns:
        List of all resulting MonoCut objects
    """
    result = []
    helper_func = partial(split_monocut_at_pauses, pause_threshold=pause_threshold)
    n_monocuts = len(monocuts)
    monocuts = iter(monocuts)
    for elem in tqdm(parallel_map(helper_func, monocuts, num_jobs=num_jobs),
        total=n_monocuts,
        desc="Splitting cuts using forced alignment",
    ):
        result.extend(elem)
    return lhotse.CutSet(result)


class TransitionType(Enum):
    """Four types of utterance transitions as defined in the paper"""

    TURN_HOLD = "TH"  # Same speaker with pause
    TURN_SWITCH = "TS"  # Different speaker with gap
    INTERRUPTION = "IR"  # Different speaker with overlap
    BACKCHANNEL = "BC"  # Different speaker fully overlapped


@dataclass
class TransitionParams:
    """Parameters for each transition type"""

    beta_th: float = 0.57  # Expected pause duration for turn-hold
    beta_ts: float = 0.40  # Expected gap duration for turn-switch
    beta_ir: float = 0.44  # Expected overlap ratio for interruption
    beta_bc: float = 0.67  # Expected overlap ratio for backchannel # not used now

    # Probability distributions
    p_ind: List[float] = None  # [p_TH, p_TS, p_IR, p_BC] for random selection
    p_markov: np.ndarray = None  # 4x4 transition matrix for Markov selection

    def __post_init__(self):
        if self.p_ind is None:
            # Default probabilities from CALLHOME1 (Table 1 in paper)
            self.p_ind = [0.15, 0.21, 0.44, 0.20]

        if self.p_markov is None:
            # Default Markov transition matrix from CALLHOME1
            self.p_markov = np.array(
                [
                    [0.26, 0.11, 0.09, 0.31],  # TH -> [TH, TS, IR, BC]
                    [0.23, 0.38, 0.29, 0.29],  # TS -> [TH, TS, IR, BC]
                    [0.27, 0.31, 0.33, 0.31],  # IR -> [TH, TS, IR, BC]
                    [0.24, 0.20, 0.29, 0.09],  # BC -> [TH, TS, IR, BC]
                ]
            ).T  # need to take transpose here

    def fit(self, supervisions: Optional[SupervisionSet] = None):
        raise NotImplementedError  # TODO this would be cool, fit on supervisionset
        pass


class ConversationalMeetingSimulator:
    def __init__(
        self,
        output_dir,
        all_cuts,
        transition_params: TransitionParams = None,
        speed_perturb=False,
        sample_rate=16000,
        use_markov=True,
        max_utt_duration=30,
        min_utt_duration=0.2,
        min_spk_utt=5,
        target_duration=120,
        min_max_spk=(2, 11),
        rirs=None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.params = transition_params or TransitionParams()
        self.sample_rate = sample_rate
        self.speed_perturb = speed_perturb
        self.use_markov = use_markov
        self.epsilon = 0.03  # For truncated exponential distribution
        self.target_duration = target_duration
        self.min_max_spk = min_max_spk
        self.rirs = rirs

        # map speakers to cuts
        logger.info("Filtering source Cuts: removing too short or too long.")
        prev_len = len(all_cuts)
        after_len = 0
        spk2cuts = {}
        for cut in all_cuts:
            if cut.duration > max_utt_duration or cut.duration < min_utt_duration:
                continue
            c_spk = cut.supervisions[0].speaker
            assert (
                len(set([x.speaker for x in cut.supervisions])) == 1
            ), "Input cuts should contain only one speaker. Yours do not."
            if c_spk not in spk2cuts.keys():
                spk2cuts[c_spk] = []
            spk2cuts[c_spk].append(cut)
            after_len += 1
        logger.info("Filtering complete.")
        logger.info(f"Before {prev_len}, now {after_len} cuts.")

        logger.info("Removing speakers with too few utterances.")
        prev_spk = len(spk2cuts.keys())
        for spk in spk2cuts.keys():
            if len(spk2cuts[spk]) < min_spk_utt:
                del spk2cuts[spk]
        logger.info(f"Before {prev_spk}, now {len(spk2cuts.keys())} speakers.")

        self.spk2cuts = spk2cuts
        self.speakers = list(spk2cuts.keys())

        # Validate Markov matrix
        if self.use_markov:
            row_sums = np.sum(self.params.p_markov, axis=1)

            if not np.allclose(row_sums, 1.0):
                logger.warning("Markov matrix rows don't sum to 1, normalizing...")
                self.params.p_markov = self.params.p_markov / (row_sums[:, None] + 1e-8)

    @staticmethod
    def sample_src_pos(l, w, mic_pos):

        def distance(current_points, ref_point):
            # current_points = np.array(current_points)
            # ref_point = np.array(ref_point)
            return np.sqrt(np.sum((current_points - ref_point) ** 2, axis=0))

        while True:
            src_x = np.random.uniform(DELTA_DIST, l - DELTA_DIST)
            src_y = np.random.uniform(DELTA_DIST, w - DELTA_DIST)
            src_z = np.random.uniform(0.5, 2)
            c_src_pos = np.array([src_x, src_y, src_z])
            if distance(c_src_pos, mic_pos) > 1:
                return c_src_pos

    @staticmethod
    def gen_rirs(meeting_id, output_dir, n_positions=30, samplerate=16000):
        # sample some amount of RIRs for each meeting and iterate over these.
        # so that speaker can change position
        while True:
            l = np.random.uniform(*ROOM_SZ)
            w = np.random.uniform(l / 2, ROOM_SZ[-1])
            z = np.random.uniform(*ROOM_CEILING)
            room_dim = [l, w, z]
            rt60 = np.random.uniform(*RT60)  # sample random RT60
            try:
                e_absorption, max_order = pyroomacoustics.inverse_sabine(rt60, room_dim)
                # retry till we can generate a valid room. This happens when rt60 is too low for a huge room.
            except ValueError:
                continue

            room = pyroomacoustics.ShoeBox(
                room_dim,
                fs=samplerate,
                max_order=max_order,
                materials=pyroomacoustics.Material(e_absorption),
                use_rand_ism=True,
                max_rand_disp=RAN_DISP,
            )
            break

        mic_x = np.random.uniform(DELTA_DIST, l - DELTA_DIST)
        mic_y = np.random.uniform(DELTA_DIST, w - DELTA_DIST)
        mic_z = np.random.uniform(0.5, z - DELTA_DIST)
        # add microphone in the room randomly

        mic_locs = np.array([mic_x, mic_y, mic_z])
        room.add_microphone(mic_locs)

        for src in range(n_positions):
            room.add_source(
                ConversationalMeetingSimulator.sample_src_pos(l, w, mic_locs)
            )

        room.compute_rir()
        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(exist_ok=True, parents=True)
        maxlen = max([len(room.rir[0][x]) for x in range(n_positions)])

        output_rirs = []
        for src in range(n_positions):
            c_rir = room.rir[0][src]
            if c_rir.shape[0] < maxlen:
                c_rir = np.pad(c_rir, (0, maxlen - c_rir.shape[0]))
            output_rirs.append(c_rir)

        output_rirs = np.stack(output_rirs).T
        flac_path = output_dir / Path(meeting_id + ".wav")
        sf.write(
            flac_path,
            output_rirs,
            samplerate=samplerate,
        )

        # Create Lhotse Recording object
        # Calculate duration from the audio data
        duration = maxlen / samplerate
        # recording will be used in the reverberate cuts function.
        # Create the recording with multichannel support
        recording = Recording(
            id=meeting_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=list(
                        range(n_positions)
                    ),  # All channels (0 to n_positions-1)
                    source=str(flac_path),
                )
            ],
            sampling_rate=samplerate,
            num_samples=maxlen,
            duration=duration,
        )

        return recording

    def sample_exponential_duration(self, beta: float) -> float:
        """Sample duration from exponential distribution"""
        return np.random.exponential(beta)

    def sample_overlap_ratio(self, beta: float) -> float:
        """Sample overlap ratio from truncated exponential distribution"""
        # Sample from exponential and truncate to [epsilon, 1-epsilon]
        ratio = np.random.exponential(beta)
        return np.clip(ratio, self.epsilon, 1.0 - self.epsilon)

    def get_offset(self, current_cut, prev_cut, prev_offset, transition_type):

        next_duration = current_cut.duration
        prev_duration = prev_cut.duration

        # if (
        #    transition_type == TransitionType.BACKCHANNEL
        #    and not next_duration <= prev_duration
        # ):
        # override as we cannot back-channel in this instance.
        # with FA data this should not be too frequent, but yeah will
        # reduce back-channels.
        #    transition_type = TransitionType.INTERRUPTION

        if transition_type == TransitionType.TURN_HOLD:
            pause_duration = self.sample_exponential_duration(self.params.beta_th)
            return prev_offset + prev_cut.duration + pause_duration

        elif transition_type == TransitionType.TURN_SWITCH:
            pause_duration = self.sample_exponential_duration(self.params.beta_ts)
            return prev_offset + prev_cut.duration + pause_duration

        elif transition_type == TransitionType.INTERRUPTION:
            overlap_ratio = self.sample_overlap_ratio(self.params.beta_ir)
            overlap_duration = overlap_ratio * prev_duration
            # start_offset = np.random.uniform(0, prev_duration)

            return prev_offset + prev_cut.duration - overlap_duration

        elif transition_type == TransitionType.BACKCHANNEL:

            # overlap_ratio = self.sample_overlap_ratio(self.params.beta_bc)
            start_offset = np.random.uniform(0, prev_duration)
            # start_offset = overlap_ratio * prev_duration

            return prev_offset + prev_cut.duration - start_offset

    def select_transition_type(
        self, prev_transition: Optional[TransitionType] = None
    ) -> TransitionType:
        """Select next transition type based on random or Markov selection"""
        if self.use_markov and prev_transition is not None:
            # Use Markov chain
            prev_idx = list(TransitionType).index(prev_transition)
            probs = self.params.p_markov[prev_idx]
        else:
            # Use independent random selection
            probs = self.params.p_ind

        # Sample transition type
        transition_idx = np.random.choice(len(TransitionType), p=probs)
        return list(TransitionType)[transition_idx]

    def create_fir_highpass(self, cutoff_freq, num_taps=101, window="hamming"):
        """
        Create a FIR highpass filter using the window method.

        Parameters:
        cutoff_freq (float): Cutoff frequency in Hz
        sample_rate (float): Sample rate in Hz
        num_taps (int): Number of filter taps (filter length). Should be odd.
        window (str): Window function ('hamming', 'hann', 'blackman', etc.)

        Returns:
        numpy.ndarray: FIR filter coefficients
        """
        sample_rate = self.sample_rate
        # Normalize the cutoff frequency (0 to 1, where 1 is Nyquist frequency)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # Create lowpass filter first
        lowpass_coeffs = firwin(num_taps, normalized_cutoff, window=window)

        # Convert to highpass by spectral inversion
        highpass_coeffs = -lowpass_coeffs
        highpass_coeffs[num_taps // 2] += 1  # Add impulse at center

        return highpass_coeffs

    def add_gaussian_noise(self, audio, min_speech_level_db, range_db_offset=(-15, 3)):
        """
        Add Gaussian noise to audio signal. Noise level is capped at 5 dB below
        the minimum speech level.

        Args:
            audio: Input audio signal (numpy array)
            min_speech_level_db: Minimum speech level in dB - noise won't exceed this by more than 5 dB

        Returns:
            Audio signal with added noise
        """
        if len(audio) == 0:
            return audio

        # Calculate maximum allowed noise level (5 dB below min speech level)
        max_noise_level_db = min_speech_level_db + np.random.uniform(*range_db_offset)

        # Generate random noise level between 0 and max allowed
        # Using a range that gives reasonable noise levels
        noise_level_db = np.random.uniform(max_noise_level_db - 20, max_noise_level_db)
        noise_rms = 10 ** (noise_level_db / 20.0)

        # Generate Gaussian noise
        noise = np.random.normal(0, noise_rms, audio.shape)

        # Add noise to signal
        noisy_audio = audio + noise

        return noisy_audio

    def normalize_to(self, audio, target_level_db):
        """
        Normalize audio to a target RMS level in dB.

        Args:
            audio: Input audio signal (numpy array)
            target_level_db: Target RMS level in dB (can be single value or array)

        Returns:
            Normalized audio signal
        """
        if len(audio) == 0:
            return audio

        # Calculate current peak RMS level
        rms = np.sqrt(np.mean(audio**2))

        # Avoid division by zero
        if rms == 0:
            return audio

        # Convert target level from dB to linear scale
        # Assuming 0 dB corresponds to RMS = 1.0
        if isinstance(target_level_db, (list, np.ndarray)):
            target_level_db = np.random.choice(target_level_db)

        target_rms_linear = 10 ** (target_level_db / 20.0)

        # Calculate gain needed
        gain = target_rms_linear / rms

        # Apply gain
        normalized_audio = audio * gain

        return normalized_audio

    def gen_audio(self, cutsets):

        n_speakers = np.random.randint(*self.min_max_spk)
        target_dur = self.target_duration

        sampled_spk = np.random.choice(self.speakers, n_speakers, replace=False)
        current_time = 0
        utt_indx = 0
        prev_transition = None
        prev_cut = None
        prev_speaker = None
        seen_speakers = set()

        utterances = []
        offsets = []

        # return [None], [None]

        fir_highpass = self.create_fir_highpass(60, 63)

        if self.rirs is not None:
            c_room_rirs = self.rirs[np.random.randint(0, len(self.rirs))]

        while current_time < target_dur or len(seen_speakers) < len(sampled_spk):

            transition_type = self.select_transition_type(prev_transition)
            if transition_type == TransitionType.TURN_HOLD:
                # Same speaker
                if prev_speaker is not None:
                    current_speaker = prev_speaker
                else:
                    # random choice of next speaker first utterance
                    current_speaker = np.random.choice(sampled_spk)
            else:
                # Different speaker
                if prev_speaker is not None:
                    available_speakers = [s for s in sampled_spk if s != prev_speaker]
                    current_speaker = np.random.choice(available_speakers)
                else:
                    current_speaker = np.random.choice(sampled_spk)

            cut = np.random.choice(self.spk2cuts[current_speaker])

            # Load audio
            # if self.speed_perturb:
            # keep these fixed
            #    factor = np.random.uniform(SPEED_PERTURB_LOW, SPEED_PERTURB_HIGH)
            #    cut = cut.perturb_speed(factor)
            # if self.rirs is not None:
            #   cut = cut.reverb_rir(np.random.choice(c_room_rirs))

            if utt_indx == 0:
                offsets.append(0.0)
                utterances.append(cut)
            else:
                c_offset = self.get_offset(cut, prev_cut, prev_offset, transition_type)
                offsets.append(c_offset)
                utterances.append(cut)

            utt_indx += 1
            prev_cut = cut
            prev_speaker = cut.supervisions[0].speaker
            seen_speakers.add(prev_speaker)
            prev_transition = transition_type
            prev_offset = offsets[-1]
            current_time = prev_offset + cut.duration

        base_gain = np.random.uniform(-35, -8)
        speech_lvls = []
        for utt in utterances:
            c_gain = np.random.uniform(-12, +5)
            speech_lvls.append(base_gain + c_gain)

        # fetch the maximum length
        output_audio = np.zeros((1, int(current_time * self.sample_rate)))

        for cut, offset, c_speech_lvl in zip(utterances, offsets, speech_lvls):
            # load audio here
            c_audio = cut.load_audio()
            initial_duration = c_audio.shape[-1]
            # remove dc offset via highpass filtering here.
            # remove anything under 65 Hz to avoid recognizing speaker from artifacts in recording
            assert c_audio.shape[0] == 1
            # this reduces overfitting on low freq noise e.g. distinguishing speakers by noise
            c_audio = c_audio - np.mean(c_audio, -1, keepdims=True)
            c_audio = fftconvolve(c_audio, fir_highpass[None, :], mode="same")
            # reverberate
            c_rir = np.random.choice(c_room_rirs)
            c_rir = c_rir.load_audio()

            peak = np.argmax(np.abs(c_rir), -1)
            around_peak = int(
                0.05 / (1 / 16000)
            )  # consider prev reflections 50ms before peak too

            try:
                c_rir = c_rir[
                    :, max(0, peak - around_peak) :
                ]  # take values around peak
            except TypeError:
                c_rir = c_rir[:, max(0, peak[0] - around_peak) :]

            # when rir is too long wrt audio then it creates artifacts
            if c_audio.shape[-1] / 2 > c_rir.shape[-1]:
                c_audio = scipy.signal.convolve(c_audio, c_rir, mode="same")
            else:
                c_rir = c_rir[..., : c_audio.shape[-1] // 4]
                c_audio = scipy.signal.convolve(c_audio, c_rir, mode="same")

            # else:
            #    pass
            # truncate rir
            # c_rir = c_rir[..., :c_audio.shape[-1]//4]

            # do not convolve for very short utterances
            # if c_rir.shape[-1] > c_audio.shape[-1]//8:

            # c_audio = np.pad(c_audio, ((0, 0), (c_rir.shape[-1], c_rir.shape[-1])), mode='constant')
            # c_audio = c_audio[:, c_rir.shape[-1]:]
            # gain adjust
            c_audio = self.normalize_to(c_audio, c_speech_lvl)
            offset = int(offset * self.sample_rate)
            maxlen = output_audio.shape[-1]
            if (offset + c_audio.shape[-1]) > maxlen:
                residual = (offset + c_audio.shape[-1]) - maxlen
                output_audio[:, offset : offset + c_audio.shape[-1]] += c_audio[
                    :, :-residual
                ]
                output_audio = np.concatenate(
                    (output_audio, c_audio[:, -residual:]), axis=-1
                )
            else:
                output_audio[:, offset : offset + c_audio.shape[-1]] += c_audio

        # add some gaussian noise here.
        # if we have noise we can add that too. e.g. wham and sins, qut etc
        min_lvl = min(speech_lvls)
        output_audio = self.add_gaussian_noise(output_audio, min_lvl, (-30, 3))

        maxval = np.amax(np.abs(output_audio))
        if maxval > 0.99:
            output_audio = output_audio * (0.99 / maxval)

        # Generate unique filename
        recording_id = str(uuid4())
        audio_filename = f"{recording_id}.wav"
        Path(self.output_dir).mkdir(exist_ok=True)
        audio_path = os.path.join(self.output_dir, audio_filename)

        # Save audio to disk
        sf.write(audio_path, output_audio.T, self.sample_rate)

        # Create Lhotse Recording
        recording = Recording(
            id=recording_id,
            sources=[AudioSource(type="file", channels=[0], source=audio_path)],
            sampling_rate=self.sample_rate,
            num_samples=output_audio.shape[-1],
            duration=output_audio.shape[-1] / self.sample_rate,
        )

        # Create Lhotse Supervisions for each utterance
        supervisions = []
        for i, (cut, offset) in enumerate(zip(utterances, offsets)):
            orig_supervision = cut.supervisions[0]

            # Adjust alignments if they exist
            alignment = None
            """
            if (
                hasattr(orig_supervision, "alignment")
                and orig_supervision.alignment is not None
            ):

                adjusted_alignment = []

                for align_item in orig_supervision.alignment["word"]:
                    # Adjust start and end times by adding the offset
                    adjusted_item = AlignmentItem(
                        symbol=align_item.symbol,
                        start=align_item.start + offset,
                        duration=align_item.duration,
                        score=getattr(align_item, "score", None),
                    )
                    adjusted_alignment.append(adjusted_item)
                alignment = adjusted_alignment
            """
            supervision = SupervisionSegment(
                id=f"{recording_id}_{i:04d}",
                recording_id=recording_id,
                start=offset,
                duration=cut.duration,
                channel=0,
                speaker=orig_supervision.speaker,
                text=getattr(orig_supervision, "text", ""),  # Use text if available
                language=getattr(orig_supervision, "language", None),
                # alignment={"word": alignment},  # Include adjusted alignments
                custom={
                    "transition_type": transition_type.name if i > 0 else "FIRST",
                    "speech_level_db": speech_lvls[i],
                    "original_cut_id": cut.id,
                },
            )
            supervisions.append(supervision)

        return recording, supervisions

    def get_mixture(self, meeting_indx):

        n_speakers = np.random.randint(*self.min_max_spk)
        target_dur = self.target_duration
        sampled_spk = np.random.choice(self.speakers, n_speakers, replace=False)
        current_time = 0
        utt_indx = 0
        prev_transition = None
        prev_cut = None
        prev_speaker = None
        seen_speakers = set()

        utterances = []
        offsets = []

        if self.rirs is not None:
            c_room_rirs = self.rirs[np.random.randint(0, len(self.rirs))]

        while current_time < target_dur or len(seen_speakers) < len(sampled_spk):

            transition_type = self.select_transition_type(prev_transition)
            if transition_type == TransitionType.TURN_HOLD:
                # Same speaker
                if prev_speaker is not None:
                    current_speaker = prev_speaker
                else:
                    # random choice of next speaker first utterance
                    current_speaker = np.random.choice(sampled_spk)
            else:
                # Different speaker
                if prev_speaker is not None:
                    available_speakers = [s for s in sampled_spk if s != prev_speaker]
                    current_speaker = np.random.choice(available_speakers)
                else:
                    current_speaker = np.random.choice(sampled_spk)

            cut = np.random.choice(self.spk2cuts[current_speaker])

            # Load audio
            if self.speed_perturb:
                # keep these fixed
                factor = np.random.uniform(SPEED_PERTURB_LOW, SPEED_PERTURB_HIGH)
                cut = cut.perturb_speed(factor)

            if self.rirs is not None:
                cut = cut.reverb_rir(np.random.choice(c_room_rirs))

            if utt_indx == 0:
                offsets.append(0.0)
                utterances.append(cut)
            else:
                c_offset = self.get_offset(cut, prev_cut, prev_offset, transition_type)

                offsets.append(c_offset)
                utterances.append(cut)

            utt_indx += 1
            prev_cut = cut
            prev_speaker = cut.supervisions[0].speaker
            seen_speakers.add(prev_speaker)
            prev_transition = transition_type
            prev_offset = offsets[-1]
            current_time = prev_offset + cut.duration

        utterances, offsets = zip(*sorted(zip(utterances, offsets), key=lambda x: x[1]))
        spk_tracks = defaultdict(list)
        for utt, offset in zip(utterances, offsets):
            spk_tracks[utt.supervisions[0].speaker].append((utt, offset))

        tracks = []
        for spk, spk_utts in spk_tracks.items():
            track, start = spk_utts[0]
            for utt, offset in spk_utts[1:]:
                track = mix(
                    track,
                    utt,
                    offset=add_durations(
                        offset, -start, sampling_rate=self.sample_rate
                    ),
                    allow_padding=True,
                )
            track = MixTrack(cut=track, type=type(track), offset=start)
            tracks.append(track)

        # sort tracks by track offset
        tracks = sorted(tracks, key=lambda x: x.offset)
        return MixedCut(id=str(uuid4()), tracks=tracks)


def split_recording_by_channels(recording: Recording) -> List[MonoCut]:
    """
    Split a multi-channel Recording into separate single-channel Cuts.
    Args:
        recording: Recording with multi-channel AudioSource
    Returns:
        List of Cuts, one per channel
    """
    # Get the multi-channel source
    source = recording.sources[0]
    channels = source.channels
    single_channel_cuts = []

    for channel in channels:
        # Create Cut for this specific channel
        channel_cut = MonoCut(
            id=f"{recording.id}_ch{channel}",
            start=0,
            duration=recording.duration,
            channel=channel,  # This tells Lhotse which channel to extract
            recording=recording,  # Reference to the original multi-channel recording
        )
        single_channel_cuts.append(channel_cut)

    return single_channel_cuts


if __name__ == "__main__":
    import sys
    import random
    num_from = int(sys.argv[1])
    num_to = int(sys.argv[2])

    np.random.seed(num_from + random.randint(0, num_to-num_from))
    random.seed(num_from + random.randint(0, num_to-num_from))

    # get cutset from librispeech.
    # split it.
    STAGE = 4
    dset_name = "librispeech"
    OUTPUT_DIR = "/scratch.ssd/dklement/job_11306482.pbs-m1/diar_data/sim_librispeech_3k_120s_correct"
    LIBRISPEECH_DIR = "/scratch.ssd/dklement/job_11306482.pbs-m1/data/librispeech"
    # LIBRISPEECH_ALIGN_DIR = "/raid/users/popcornell/ESPNet3/espnet/egs3/ami/diar1/data/librispeech_align/LibriSpeech"
    # path to librispeech alignment from lhotse
    LIBRISPLITs = ["train-clean-100", "train-clean-360", "train-other-500"]
    N_POSITIONS_RIRs = 50
    SPLIT_FA_FACTOR = 0.1
    N_RIRs = 10000
    NUM_MEETINGS = num_to - num_from  # 3k hours for 120 seconds meetings
    # 1000 meetings I can do in 5 mins with 16 jobs.
    N_JOBs = 1
    MIN_MAX_SPK = (2, 11)
    DURATION = 120

    if STAGE <= 0:
        print("Stage 0...")
        lhotse_manifest_dir = os.path.join(LIBRISPEECH_DIR, "librispeech_manifests")
        # Path(lhotse_manifest_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(OUTPUT_DIR, "librispeech_manifests")).mkdir(parents=True, exist_ok=True)

        # prepare_librispeech(
        #     corpus_dir=LIBRISPEECH_DIR,
        #     alignments_dir=LIBRISPEECH_ALIGN_DIR,
        #     output_dir=os.path.join(OUTPUT_DIR, "librispeech_manifests"),
        #     dataset_parts=LIBRISPLITs,
        #     num_jobs=N_JOBs,
        # )

        all_cuts = []
        for split in LIBRISPLITs:
            c_rec = lhotse.load_manifest(
                Path(lhotse_manifest_dir) / f"librispeech_recordings_{split}.jsonl.gz"
            )
            c_sup = lhotse.load_manifest(
                Path(lhotse_manifest_dir) / f"librispeech_supervisions_{split}.jsonl.gz"
            )
            c_cut = CutSet.from_manifests(recordings=c_rec, supervisions=c_sup)
            all_cuts.append(c_cut)
        all_cuts = combine_manifests(all_cuts)
        logger.info("Saving source CutSet to disk")
        all_cuts.to_file(
            os.path.join(OUTPUT_DIR, "librispeech_manifests", "all_cuts.jsonl.gz")
        )

    if STAGE <= 2:
        print("Stage 2...")
        try:
            all_cuts
        except NameError:
            logger.info("Loading source CutSet from disk")
            all_cuts = lhotse.load_manifest(
                os.path.join(OUTPUT_DIR, "librispeech_manifests", "all_cuts.jsonl.gz")
            )
        logger.info(f"Before splitting with forced alignment: {len(all_cuts)} cuts.")
        all_cuts = split_monocuts_batch(all_cuts, SPLIT_FA_FACTOR, num_jobs=N_JOBs)
        logger.info(f"After splitting with forced alignment: {len(all_cuts)} cuts.")
        logger.info(f"Saving to disk splitted cuts.")
        all_cuts.to_file(
            os.path.join(
                OUTPUT_DIR, "librispeech_manifests", "all_cuts_splitted.jsonl.gz"
            )
        )

    if STAGE <= 3:
        print("Stage 3...")
        logger.info("Simulating RIRs using Pyroomacoustics")

        helper_func = partial(
            ConversationalMeetingSimulator.gen_rirs,
            output_dir=os.path.join(OUTPUT_DIR, "rirs"),
            n_positions=N_POSITIONS_RIRs,
        )

        meeting_ids = iter([f"rirs_{x}" for x in range(N_RIRs)])
        all_rirs = []
        for recording in tqdm(
            parallel_map(helper_func, meeting_ids, num_jobs=N_JOBs),
            total=N_RIRs,
            desc="Simulating room impulse responses (RIRs)",
        ):
            all_rirs.append(recording)
        all_rirs = RecordingSet.from_recordings(all_rirs)
        all_rirs.to_file(os.path.join(OUTPUT_DIR, "rirs", "rirs.json.gz"))


    if STAGE <= 4:
        try:
            all_cuts
        except NameError:
            logger.info("Loading source CutSet")
            all_cuts = lhotse.load_manifest(
                os.path.join(
                    OUTPUT_DIR, "librispeech_manifests", "all_cuts_splitted.jsonl.gz"
                )
            )
            logger.info("Source CutSet loaded")

        rirs = lhotse.load_manifest(os.path.join(OUTPUT_DIR, "rirs", "rirs.json.gz"))
        tmp = []
        for recording in rirs:
            c_set = split_recording_by_channels(recording)
            tmp.append(CutSet.from_cuts(c_set))
        rirs = tmp

        simulator = ConversationalMeetingSimulator(
            Path(OUTPUT_DIR).absolute() / Path("audio"),
            all_cuts,
            rirs=rirs,
            min_max_spk=MIN_MAX_SPK, target_duration=DURATION
        )

        output_dir = Path(OUTPUT_DIR).absolute() / Path("manifests")
        output_dir.mkdir(exist_ok=True, parents=True)

        uuids = iter([f"simulation_{x}" for x in range(num_from, num_to)])
        mixtures = []
        work = partial(simulator.gen_audio)

        recordings = []
        supervisions = []
        # for c_rec, c_sup in tqdm(
        #     parallel_map(work, uuids, num_jobs=N_JOBs, threads=True),
        #     total=NUM_MEETINGS,
        #     desc="Simulating meetings",
        # ):
        for c_rec, c_sup in tqdm(map(simulator.gen_audio, uuids), total=NUM_MEETINGS, desc="Simulating meetings"):
            recordings.append(c_rec)
            supervisions.extend(c_sup)

        supervisions = SupervisionSet(supervisions)
        recordings = RecordingSet(recordings)
        lhotse.validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        logger.info("Saving simulated manifests to disk")
        supervisions.to_file(output_dir / f"sim-{dset_name}-train-supervisions-{num_from}-{num_to}.jsonl.gz")
        recordings.to_file(output_dir / f"sim-{dset_name}-train-recordings-{num_from}-{num_to}.jsonl.gz")

        cutset = lhotse.CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        # save also cutset here.
        cutset.to_file(output_dir / f"sim-{dset_name}-train-cuts-{num_from}-{num_to}.jsonl.gz")

    """
    if STAGE <= 4:
        output_dir = Path(OUTPUT_DIR).absolute() / Path("simulated")
        logger.info("Dumping recordings to disk. Running simulation.")
        try:
            cutset
        except NameError:
            logger.info("Loading simulation CutSet")
            cutset = lhotse.load_manifest(output_dir / "simulated.json.gz")
            logger.info("Simulation CutSet loaded")

        from lhotse import CutSet
        from lhotse.features import Fbank, FbankConfig

        # Create feature extractor
        extractor = Fbank(FbankConfig(num_mel_bins=80))

        # Extract features in parallel using CutSet's built-in parallelization
        cutset = cutset.compute_and_store_features(
            extractor=extractor,
            # augment_fn=highpass_f,
            storage_path=output_dir / Path("features"),  # Directory to store features
            num_jobs=N_JOBs,  # Number of parallel processes
        )
        logger.info("Saving simulated CutSet to disk")
        cutset.to_file(output_dir / "simulated_w_features.json.gz")
    """
