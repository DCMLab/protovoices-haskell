from IPython.display import Audio, display
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fractions import Fraction as frac
import os.path as path
import pitchtypes as pt
import re

# loading dcml corpus data
# ------------------------

str2inttuple = lambda l: tuple() if l == '' else tuple(int(s) for s in l.split(', '))
def int2bool(s):
    try:
        return bool(int(s))
    except:
        return s

CONVERTERS = {
    'added_tones': str2inttuple,
    'act_dur': frac,
    'chord_tones': str2inttuple,
    'duration': frac,
    'globalkey_is_minor': int2bool,
    'localkey_is_minor': int2bool,
    'mc_offset': frac,
    'mc_onset': frac,
    'mn_onset': frac,
    'next': str2inttuple,
    'nominal_duration': frac,
    #'offset': frac,
    #'onset': frac,
    'scalar': frac,
}

STRING = 'string' # not str

DTYPES = {
    'alt_label': STRING,
    'barline': STRING,
    'bass_note': 'Int64',
    'breaks': STRING,
    #'cadence': STRING,
    #'cadences_id': 'Int64',
    'changes': STRING,
    'chord': STRING,
    'chord_id': int,
    'chord_type': STRING,
    'dont_count': 'Int64',
    'figbass': STRING,
    'form': STRING,
    'globalkey': STRING,
    #'gracenote': STRING,
    #'harmonies_id': 'Int64',
    'keysig': int,
    'label': STRING,
    'localkey': STRING,
    'mc': int,
    'midi': int,
    'mn': int,
    #'notes_id': 'Int64',
    'numbering_offset': 'Int64',
    'numeral': STRING,
    'pedal': STRING,
    #'playthrough': int,
    'phraseend': STRING,
    'relativeroot': STRING,
    'repeats': STRING,
    'root': 'Int64',
    'special': STRING,
    'staff': int,
    'tied': 'Int64',
    'timesig': STRING,
    'tpc': int,
    'voice': int,
    #'voices': int,
    'volta': 'Int64'
}

def load_dcml_tsv(corpusdir, piece, kind):
    filename = path.join(corpusdir, kind, piece + '.' + kind + '.tsv')
    return pd.read_csv(filename, sep='\t', converters=CONVERTERS, dtype=DTYPES)

def name2tpc(name):
    pitch = name[0].upper() + name[1:]
    return pt.SpelledPitchClass(pitch).fifths()


r_numeral = re.compile(
    r"(?P<acc>b*|\#*)(?P<num>VII|VI|V|IV|III|II|I)",
    re.IGNORECASE
)

NUMERALS = {
    "i": 0,
    "ii": 1,
    "iii": 2,
    "iv": 3,
    "v": 4,
    "vi": 5,
    "vii": 6,
}

MAJOR = [pt.SpelledIntervalClass(s)
         for s in ["P1", "M2", "M3", "P4", "P5", "M6", "M7"]]

MINOR = [pt.SpelledIntervalClass(s)
         for s in ["P1", "M2", "m3", "P4", "P5", "m6", "m7"]]


def numeral2tic(label, major):
    m = r_numeral.match(label)
    # accidentals
    accs = m.group("acc")
    if len(accs) == 0:
        mod = pt.SpelledIntervalClass.unison()
    elif accs[0] == '#':
        mod = pt.SpelledIntervalClass.chromatic_semitone() * len(accs)
    else:
        mod = - pt.SpelledIntervalClass.chromatic_semitone() * len(accs)
    # numeral
    degree = NUMERALS[m.group("num").lower()]
    interval = MAJOR[degree] if major else MINOR[degree]
    return (interval + mod).fifths()

# data preprocessing
# ------------------

def isneighbour(p1,p2):
    """
    Returns True if p1 and p2 could be neighbors,
    that is, if the interval between them is either a chromatic semitone or a generic second.
    """
    return (abs(p1-p2)%7 in [2,5]) or (abs(p1-p2) == 7)

def notetype(pitch, pitches, chordtones):
    """
    Returns the estimated type of an observed pitch ('chordtone', 'ornament', or 'unknown')
    given the set of all observed pitches and the set of nominal chord tones.

    A pitch is an actual chord tone with certainty iff it either does not have any neighbor
    or it is a nominal chord tone (part of the prototype) and does not have any nominal chord tone neighbors.
    A pitch is an ornament with certainty
    iff it is the neighbor of a nominal chord tone but is not a nominal chord tone itself.
    Otherwise, the type is considered to be unknown.
    """
    hasneighbour = any(isneighbour(p, pitch) for p in pitches) # any uses iterator here
    hasctneighbour = any(isneighbour(p, pitch) and (p in chordtones) for p in pitches) 
    
    if not hasneighbour: # if there is no identifiable neighbour, tone has to be chord tone
        return 'chordtone'
    elif pitch in chordtones and not hasctneighbour:
        return 'chordtone'
    elif not pitch in chordtones and hasctneighbour:
        return 'ornament'
    else:
        return 'unknown'

# pitch class range
# -----------------

fifth_range = 2*7

def set_fifth_range(f):
    global fifth_range
    fifth_range = f

def get_fifth_range():
    return fifth_range

def get_npcs():
    return 2*fifth_range + 1

def fifth_to_index(fifth):
    """Turns a LoF pitch class into an index."""
    return fifth + fifth_range

def index_to_fifth(index):
    """Turns an index into a LoF pitch class"""
    return index - fifth_range

def chord_tensor(fifths, types, device="cpu"):
    """Takes a list of notes as fifths and a list of corresponding note types."""
    notetype = {'chordtone': 0, 'ornament': 1, 'unknown': 2}
    chord = torch.zeros((3, get_npcs()), device=device)
    for (fifth, t) in zip(fifths, types):
        chord[notetype[t], fifth_to_index(fifth)] += 1
    return chord.reshape((1,-1))

# loading data
# ------------

def load_csv(fn, sep='\t'):
    df = pd.read_csv(fn, sep=sep).dropna()
    # some notes are too far away from the root for our model,
    # so we wrap them to their closest enharmonic equivalent
    df['fifth'] = wrap_fifths(df.fifth.astype(int))
    return df

def wrap_fifths(fifths):
    fifths = fifths.copy()
    too_high = (fifths > fifth_range)
    too_low = (fifths < -fifth_range)

    fth = fifths[too_high]
    fifths.loc[too_high] = fth - (np.ceil((fth - fifth_range) / 12.).astype(int) * 12)

    ftl = fifths[too_low]
    fifths.loc[too_low] = ftl + (np.ceil(-(ftl + fifth_range) / 12.).astype(int) * 12)
    return fifths

# plotting etc.
# -------------

def play_chord(amplitudes, T=2, sr=22050):
    # tone frequencies based on line of fifths
    tpcs = np.arange(-fifth_range, fifth_range+1)
    tones = 262. * np.exp(tpcs * np.log(1.5) % np.log(2))

    # # tone frequencies based on 12TET
    # npcs = tpcs * 7 % 12
    # tones = 262 * np.exp(npcs * np.log(2) / 12)

    n = int(T*sr)
    t = np.linspace(0, T, n)
    sines = np.sin(2 * np.pi * tones[:, np.newaxis] * t[np.newaxis, :])
    amps = amplitudes / amplitudes.sum()

    # # add octave above and below
    # tones = np.concatenate((0.5*tones, tones, 2*tones))
    # amps = amps.repeat(3)

    mix = 0.1 * np.dot(amps, sines) # ear safety first
    display(Audio(mix, rate=sr, normalize=False))


def plot_profile(chordtones, ornaments, name):
    labels = np.arange(-fifth_range, fifth_range+1)
    x = np.arange(get_npcs())
    width = 0.4
    fig, ax = plt.subplots(figsize=(15,5))
    plt.bar(x - width/2, chordtones, width, label='chord tones')
    plt.bar(x + width/2, ornaments, width, label='ornaments')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show(block=False)
