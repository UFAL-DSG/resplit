import collections
import os
import wave
import logging

from fnnvad import FFNNVAD

LOGGING_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

SAMPLE_RATE = 8000
READ_BUFFER_SIZE = 128

FRAMES_PER_SECOND = SAMPLE_RATE * 2 / READ_BUFFER_SIZE

PRE_DETECTION_BUFFER_FRAMES = int(FRAMES_PER_SECOND * 0.5)
SMOOTHE_DECISION_WINDOW_SIL = int(FRAMES_PER_SECOND * 0.2)
SMOOTHE_DECISION_WINDOW_SPEECH = int(FRAMES_PER_SECOND * 0.2)
DECISION_SPEECH_THRESHOLD = 0.7
DECISION_NON_SPEECH_THRESHOLD = 0.1


vad_cfg = {
    'framesize': 512,
    'frameshift': 160,
    'sample_rate': SAMPLE_RATE,
    'usehamming': True,
    'preemcoef': 0.97,
    'numchans': 26,
    'ceplifter': 22,
    'numceps': 12,
    'enormalise': True,
    'zmeansource': True,
    'usepower': True,
    'usec0': False,
    'usecmn': False,
    'usedelta': False,
    'useacc': False,
    'n_last_frames': 30, # 15,
    'n_prev_frames': 15,
    'mel_banks_only': True,
    'lofreq': 125,
    'hifreq': 3800,
    'model': 'vad_nnt_1196_hu512_hl1_hla3_pf30_nf15_acf_4.0_mfr32000000_mfl1000000_mfps0_ts0_usec00_usedelta0_useacc0_mbo1_bs1000.tffnn',
    'filter_length': 2,
}


def convert_to_wav(in_file, out_file, chan):
    os.system('sox -e signed-integer -b 16 -r %d -c 2 -t raw "%s" "%s" remix %d' % (SAMPLE_RATE, in_file, out_file, chan, ))


def main(input_dir, output_dir, v):
    if v:
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    else:
        logging.basicConfig(format=LOGGING_FORMAT)

    logging.info('Starting.')

    if not os.path.exists(vad_cfg['model']):
        os.system('wget "%s"' % ('wget https://vystadial.ms.mff.cuni.cz/download/alex/resources/vad/voip/%s' % vad_cfg['model'], ))

    logging.info('Loading VAD model.')
    vad = FFNNVAD(**vad_cfg)

    to_process = []

    logging.info('Searching for files.')
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.pcm'):
                to_process.append((file_name, root))

    logging.info('Processing files.')
    for file_name, root in to_process:
        process_file(vad, file_name, root, output_dir)


def process_file(vad, file_name, root, out_base):
    file_path = os.path.join(root, file_name)
    out_dir = os.path.join(out_base, root, file_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    wav_path_a = os.path.join(out_dir, 'all.a.wav')
    wav_path_b = os.path.join(out_dir, 'all.b.wav')
    convert_to_wav(file_path, wav_path_a, 1)
    convert_to_wav(file_path, wav_path_b, 2)

    res_files = vad_split(vad, wav_path_a, out_dir, "a")
    res_files = vad_split(vad, wav_path_b, out_dir, "b")


def vad_split(vad, file_name, out_dir, out_prefix):
    logging.info('Splitting %s' % file_name)

    with wave.open(file_name) as wave_in:
        res_files = []
        res_file_cntr = 0

        frames = []

        is_speech = False
    
        detection_window_speech = collections.deque(maxlen=SMOOTHE_DECISION_WINDOW_SPEECH)
        detection_window_sil = collections.deque(maxlen=SMOOTHE_DECISION_WINDOW_SIL)
        pre_detection_buffer = collections.deque(maxlen=PRE_DETECTION_BUFFER_FRAMES)

        prebuffered = False

        while 1:
            if not prebuffered:
                audio_data = wave_in.readframes(READ_BUFFER_SIZE)
            else:
                prebuffered = False

            if len(audio_data) == 0:
                break



            raw_vad_decision = vad.decide(audio_data)
            is_speech = smoothe_decison(raw_vad_decision, is_speech, detection_window_speech, detection_window_sil)

            if not is_speech:
                pre_detection_buffer.append(audio_data)

            frames.append(audio_data)

            if not is_speech and len(frames) > 1:
                _save_part(res_file_cntr, list(pre_detection_buffer) + frames, out_dir, res_files, wave_in, out_prefix)
                res_file_cntr += 1
                pre_detection_buffer.extend(frames[-PRE_DETECTION_BUFFER_FRAMES:])

            if not is_speech:
                frames = []

        _save_part(res_file_cntr, frames, out_dir, res_files, wave_in, out_prefix)

    return res_files


def smoothe_decison(decision, last_vad, detection_window_speech, detection_window_sil):
    detection_window_speech.append(decision)
    detection_window_sil.append(decision)

    speech = float(sum(detection_window_speech)) / (len(detection_window_speech) + 1.0)
    sil = float(sum(detection_window_sil)) / (len(detection_window_sil) + 1.0)

    print "speech(%.2f) sil(%.2f)" % (speech, sil)

    vad = last_vad
    change = None
    if last_vad:
        # last decision was speech
        if sil < DECISION_NON_SPEECH_THRESHOLD:
            vad = False
            #change = 'non-speech'
    else:
        if speech > DECISION_SPEECH_THRESHOLD:
            vad = True
            #change = 'speech'

    return vad




def _save_part(cntr, frames, out_dir, res_files, wave_in, out_prefix):
    logging.info('Saving part %d (%d frames).' % (cntr, len(frames)))

    res_file = os.path.join(out_dir, 'part.%s.%.3d.wav' % (out_prefix, cntr, ))
    wf = wave.open(res_file, 'wb')
    wf.setnchannels(wave_in.getnchannels())
    wf.setsampwidth(wave_in.getsampwidth())
    wf.setframerate(wave_in.getframerate())
    wf.writeframes(b''.join(frames))
    wf.close()
    res_files.append(res_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-v', default=False, action='store_true')

    args = parser.parse_args()

    main(**vars(args))