import collections
import os
import wave
import logging

from fnnvad import FFNNVAD

LOGGING_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

default_vad_cfg = {
    'framesize': 512,
    'frameshift': 160,
    'sample_rate': 8000,
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


class RecordingSplitter(object):
    SAMPLE_RATE = 8000
    SAMPLE_WIDTH = 2
    READ_BUFFER_SIZE = 128
    BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH

    FRAMES_PER_SECOND = BYTES_PER_SECOND / READ_BUFFER_SIZE

    PRE_DETECTION_BUFFER_FRAMES = int(FRAMES_PER_SECOND * 0.5)
    SMOOTHE_DECISION_WINDOW_SIL = int(FRAMES_PER_SECOND * 0.2)
    SMOOTHE_DECISION_WINDOW_SPEECH = int(FRAMES_PER_SECOND * 0.2)
    DECISION_SPEECH_THRESHOLD = 0.7
    DECISION_NON_SPEECH_THRESHOLD = 0.1

    CHANGE_TO_NON_SPEECH = 2
    CHANGE_TO_SPEECH = 1


    def __init__(self, vad_cfg, sample_rate):
        assert vad_cfg['sample_rate'] == sample_rate, 'Sample rates for VAD and the recording must match!'
        self.vad_cfg = vad_cfg
        self.SAMPLE_RATE = sample_rate

        logging.info('Loading VAD model.')
        self.vad = FFNNVAD(**vad_cfg)


    def convert_to_wav(self, in_file, out_file, chan):
        os.system('sox -e signed-integer -b 16 -r %d -c 2 -t raw "%s" "%s" remix %d' % (self.SAMPLE_RATE, in_file, out_file, chan, ))

    def split_pcm(self, file_name, root, out_base):
        file_path = os.path.join(root, file_name)
        out_dir = os.path.join(out_base, root, file_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        wav_path_a = os.path.join(out_dir, 'all.a.wav')
        wav_path_b = os.path.join(out_dir, 'all.b.wav')
        self.convert_to_wav(file_path, wav_path_a, 1)
        self.convert_to_wav(file_path, wav_path_b, 2)

        res_files1 = self.split_single_channel_wav(wav_path_a, out_dir, "a")
        res_files2 = self.split_single_channel_wav(wav_path_b, out_dir, "b")

        res = res_files1 + res_files2
        res.sort(key=lambda ((tb, te, ), fn, ): tb)

        return res

    def split_single_channel_wav(self, file_name, out_dir, out_prefix):
        logging.info('Splitting %s' % file_name)

        wave_in = wave.open(file_name)
        res_files = []
        res_file_cntr = 0

        frames = []

        is_speech = False
        n_read = 0
        n_read_beg = None

        detection_window_speech = collections.deque(maxlen=self.SMOOTHE_DECISION_WINDOW_SPEECH)
        detection_window_sil = collections.deque(maxlen=self.SMOOTHE_DECISION_WINDOW_SIL)
        pre_detection_buffer = collections.deque(maxlen=self.PRE_DETECTION_BUFFER_FRAMES)

        while 1:
            audio_data = wave_in.readframes(self.READ_BUFFER_SIZE)
            n_read += self.READ_BUFFER_SIZE

            if len(audio_data) == 0:
                break

            raw_vad_decision = self.vad.decide(audio_data)
            is_speech, change = self._smoothe_decison(raw_vad_decision, is_speech, detection_window_speech, detection_window_sil)

            if not is_speech:
                pre_detection_buffer.append(audio_data)

            if change == self.CHANGE_TO_SPEECH:
                n_read_beg = n_read - self.READ_BUFFER_SIZE
                frames = []
            elif change == self.CHANGE_TO_NON_SPEECH:
                #if not is_speech and len(frames) > 1:
                self._save_part(res_file_cntr, list(pre_detection_buffer) + frames, out_dir, res_files, wave_in, out_prefix, n_read_beg, n_read)
                res_file_cntr += 1
                pre_detection_buffer.extend(frames[-self.PRE_DETECTION_BUFFER_FRAMES:])

            if is_speech:
                frames.append(audio_data)

            if res_file_cntr > 1:
                break

        self._save_part(res_file_cntr, frames, out_dir, res_files, wave_in, out_prefix, n_read_beg, n_read)

        return res_files


    def _smoothe_decison(self, decision, last_vad, detection_window_speech, detection_window_sil):
        detection_window_speech.append(decision)
        detection_window_sil.append(decision)

        speech = float(sum(detection_window_speech)) / (len(detection_window_speech) + 1.0)
        sil = float(sum(detection_window_sil)) / (len(detection_window_sil) + 1.0)

        vad = last_vad
        change = None
        if last_vad:
            # last decision was speech
            if sil < self.DECISION_NON_SPEECH_THRESHOLD:
                vad = False
                change = self.CHANGE_TO_NON_SPEECH
        else:
            if speech > self.DECISION_SPEECH_THRESHOLD:
                vad = True
                change = self.CHANGE_TO_SPEECH

        return vad, change




    def _save_part(self, cntr, frames, out_dir, res_files, wave_in, out_prefix, n_read_beg, n_read_end):
        logging.info('Saving part %d (%d frames).' % (cntr, len(frames)))

        res_file = os.path.join(out_dir, 'part.%s.%.3d.wav' % (out_prefix, cntr, ))
        wf = wave.open(res_file, 'wb')
        wf.setnchannels(wave_in.getnchannels())
        wf.setsampwidth(wave_in.getsampwidth())
        wf.setframerate(wave_in.getframerate())
        wf.writeframes(b''.join(frames))
        wf.close()

        res_files.append(((n_read_beg * 1.0 / self.BYTES_PER_SECOND, n_read_end * 1.0 / self.BYTES_PER_SECOND), res_file))



def main(input_dir, output_dir, v):
    if v:
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    else:
        logging.basicConfig(format=LOGGING_FORMAT)

    logging.info('Starting.')
    to_process = _find_files_to_split(input_dir)

    vad_cfg = default_vad_cfg
    _download_vad_model_if_not_exists(vad_cfg)

    rs = RecordingSplitter(vad_cfg=vad_cfg, sample_rate=8000)

    _split_files(rs, output_dir, to_process)


def _download_vad_model_if_not_exists(vad_cfg):
    if not os.path.exists(vad_cfg['model']):
        os.system('wget "%s"' % (
        'wget https://vystadial.ms.mff.cuni.cz/download/alex/resources/vad'
        '/voip/%s' %
        vad_cfg['model'], ))



def _find_files_to_split(input_dir):
    to_process = []
    logging.info('Searching for files.')
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith('.pcm'):
                to_process.append((file_name, root))
    return to_process


def _split_files(rs, output_dir, to_process):
    logging.info('Processing files.')
    for file_name, root in to_process:
        files = rs.split_pcm(file_name, root, output_dir)
        _create_session_xml(output_dir, files)


def _create_session_xml(output_dir, files):
    res = """<?xml version="1.0" encoding="utf-8"?>
<dialogue>
    <config>
    </config>
    <header>
        <host>{host}</host>
        <date>{date}</date>
        <system>{system}</system>
        <version>{version}</version>
        <input_source type="voip"/>
    </header>
    %s
</dialogue>
    """

    turn_tpl = """<turn speaker="user" time="{turn_time}" turn_number="{turn_num}">
        <rec starttime="{rec_starttime}" endtime="{rec_endtime}" fname="{rec_filename}" />
    </turn>"""

    res_turns = []
    for i, ((ts, te), fn) in enumerate(files):
        turn = turn_tpl.format(turn_time=ts,
                               turn_num=i + 1,
                               rec_starttime=ts,
                               rec_endtime=te,
                               rec_filename=fn)
        res_turns.append(turn)

    session_fn = os.path.join(output_dir, 'session.xml')

    with open(session_fn, 'w') as f_out:
        f_out.write(res % "\n".join(res_turns))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('-v', default=False, action='store_true')

    args = parser.parse_args()

    main(**vars(args))