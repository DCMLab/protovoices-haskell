import dataclasses
from dataclasses import dataclass
import pandas as pd
import numpy as np
from utils import load_dcml_tsv, name2tpc, numeral2tic
from pathlib import Path
import json
import tqdm
import multiprocessing as mp
from fractions import Fraction


# helper function
# ---------------


def encode_json(x):
    if isinstance(x, Fraction):
        return {'n': x.numerator, 'd': x.denominator}

    return x


def total_onsets(events: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the onsets of events (chord labels or notes)
    relative to the beginning of the piece
    by converting them from measure-relative notation.
    """
    moffsets = measures.act_dur.values.cumsum()
    monsets = moffsets - measures.act_dur.values
    mi = events.mc - 1
    return events.mc_onset + monsets[mi]


def merge_ties(notes: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of notes with ties merged.
    """
    notes = notes.copy()
    beginnings = notes[notes.tied == 1]
    continues = notes[notes.tied < 1]
    for i in beginnings.index:
        on = notes.total_onset[i]
        off = notes.total_offset[i]
        midi = notes.midi[i]
        tpc = notes.tpc[i]
        while True:
            cont = continues[(continues.total_onset == off) &
                             (continues.midi == midi) &
                             (continues.tpc == tpc)].first_valid_index()
            if cont is None:
                break
            off = continues.total_offset[cont]
            if continues.tied[cont] == -1:
                break
        notes.at[i, 'total_offset'] = off
        notes.at[i, 'duration'] = off - on
    return notes[~(notes.tied < 1).fillna(False)]


def load_dfs(corpus: str, piece: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Loads and preprocesses dataframes for the notes and chord labels of a piece.
    """
    measures = load_dcml_tsv(corpus, piece, 'measures')

    notes = load_dcml_tsv(corpus, piece, 'notes')
    notes['total_onset'] = total_onsets(notes, measures)
    notes['total_offset'] = notes.total_onset.values + notes.duration.values
    notes = merge_ties(notes)
    max_offset = notes.total_offset.values.max()

    harmonies = load_dcml_tsv(corpus, piece, 'harmonies')
    harmonies = harmonies[~harmonies.chord.isnull()]
    harmonies['total_onset'] = total_onsets(harmonies, measures)
    harmonies['total_offset'] = np.append(harmonies.total_onset.values[1:], max_offset)
    # if 'special' in harmonies.columns:
    #     harmonies['actual_chord_type'] = harmonies.special.fillna(harmonies.chord_type)
    # else:
    #     harmonies['actual_chord_type'] = harmonies.chord_type

    return notes, harmonies

# extracting chords
# -----------------


@dataclass
class Chord:
    label: str           # chord-type label
    root: int            # absolute root as tpc
    expected: list[int]  # expected chord tones as (absolute) tpc
    corpus: str          # source corpus
    piece: str           # source piece ID
    mn: int              # measure (onset)
    mn_onset: Fraction   # location in the measure (onset)
    notes: dict          # dataframe-like dictionary of notes in the chord


def get_chords(notes: pd.DataFrame,
               harmonies: pd.DataFrame,
               corpus: str,
               piece: str) -> list[Chord]:
    """
    Computes chords as label x note pairs for a piece (given by its notes and chord labels).
    Pairs that belong to the same chord get the same id, starting from id_offset.
    Returns the dataframe of chords and the highest used id.
    """
    key = name2tpc(harmonies.globalkey[0])
    global_major = not harmonies.globalkey_is_minor[0]

    # # setup the columns of the result dataframe
    # chordids = np.empty(0, dtype=int)
    # labels   = np.empty(0, dtype=str)
    # fifths   = np.empty(0, dtype=int)
    # types    = np.empty(0, dtype=str)
    # # running id counter
    # current_id = 0

    # for checking whether the chord label is empty at some point
    chord_is_null = harmonies.chord.isnull()

    chords = []

    # iterate over all harmonies
    for i, ih in enumerate(harmonies.index):
        # chord label empty or '@none'? then skip
        if chord_is_null[ih] or harmonies.chord[ih] == '@none':
            continue

        # get info about the current harmony
        on = harmonies.total_onset[ih]
        off = harmonies.total_offset[ih]
        label = harmonies.chord[ih]  # harmonies.actual_chord_type[ih]
        local_key = numeral2tic(harmonies.localkey[ih], global_major) + key
        root = harmonies.root[ih] + local_key
        expected = list(harmonies.chord_tones[ih]) + list(harmonies.added_tones[ih])
        mn = int(harmonies.mc[ih])
        mn_onset = Fraction(harmonies.mn_onset[ih])

        # compute the corresponding notes, their pitches, and their note types
        inotes = (notes.total_offset > on) & (notes.total_onset < off)
        chord_notes = notes.loc[inotes, ["total_onset", "total_offset", "tpc", "octave"]]
        # cut notes that are held over before or after the chord
        chord_notes.total_onset.clip(on, off, inplace=True)
        chord_notes.total_offset.clip(on, off, inplace=True)

        chord = Chord(
            label=label,
            root=int(root),
            corpus=corpus,
            piece=piece,
            mn=mn,
            mn_onset=mn_onset,
            expected=[pc + local_key for pc in expected],
            notes=chord_notes.to_dict(orient='list'),
        )
        chords.append(chord)

    return chords

# processing files
# ----------------


type PieceID = (Path, str)


def get_chords_from_piece(piece: PieceID) -> (list[Chord], str):
    """
    Same as get_chords, but takes a corpus subdirectory and a piece id
    """
    folder, piece_id = piece
    name = f"{folder} {piece_id}"
    try:
        notes, harmonies = load_dfs(folder, piece_id)
        chords = get_chords(notes, harmonies, folder.name, piece_id)
        return chords, name
    except FileNotFoundError:
        print(f'file not found for {name}')
        return None
    except ValueError:
        print(f'ValueError in {name}')
        return None
    except (KeyboardInterrupt):
        print("interrupted by user, exiting.")
        quit()
    except Exception as e:
        print(f'error while processing {name}:\n{e}')
        return None


def get_chords_from_files(filelist: PieceID) -> list[Chord]:
    """
    Returns the combined chords for several pieces.
    Takes a list of subdirectory x piece pairs.
    """

    # load all files and extract chords (slow, parallelized)
    with mp.Pool() as pool:
        outputs = list(tqdm.tqdm(pool.imap(get_chords_from_piece, filelist),
                                 total=len(filelist)))

    # collect results in one large list
    all_chords = []
    files = []
    for output in tqdm.tqdm(outputs):
        if output is not None:
            chords, name = output
            all_chords += chords
            files.append(name)
    total_chords = len(all_chords)

    # write log
    msg = f"Got {total_chords} chords from {len(files)} files"
    print(msg, "listed in data/preprocess_dcml.txt")
    with open(Path("preprocess_dcml.txt"), "w") as f:
        print(msg, file=f)
        f.write("\n".join(files))

    return all_chords


def get_corpus_pieces(corpus) -> list[PieceID]:
    """
    Returns a list of pieces in a corpus as subdirectory x piece pairs.
    """
    corpus = Path(corpus)
    print("fetching pieces from", corpus)
    dirs = [d.parent for d in corpus.glob('*/harmonies')]
    print(dirs)
    # sort for consistent order
    files = sorted((d, f.with_suffix('').stem)
                   for d in dirs
                   for f in d.glob('harmonies/*.harmonies.tsv'))
    return files

# script
# ------


if __name__ == "__main__":
    print("scanning corpus...")
    pieces_dcml = get_corpus_pieces(Path("dcml_corpora"))
    # pieces_romantic = get_corpus_pieces(Path("data", "romantic_piano_corpus"))
    pieces = pieces_dcml  #  + pieces_romantic
    print(len(pieces), "pieces")
    print("extracting chords from pieces...")
    all_chords = [dataclasses.asdict(chord)
                  for chord in get_chords_from_files(pieces)]
    print("writing chords...")
    # all_chords.to_csv(Path('data', 'dcml.tsv'), sep='\t', index=False)
    with open("chords.json", "w") as f:
        json.dump(all_chords, f, default=encode_json)
    print("done.")
