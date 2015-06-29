#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
from scipy.misc import logsumexp
import struct

from math import log

from tffnn import TheanoFFNN
from mfcc import MFCCFrontEnd


class FFNNVAD():
    """ This is implementation of a FFNN based voice activity detector.

    It only implements decisions whether input frame is speech of non speech.
    It returns the posterior probability of speech for N last input frames.
    """
    def __init__(self, model, filter_length, sample_rate, framesize, frameshift,
                 usehamming, preemcoef, numchans,  ceplifter, numceps,
                 enormalise, zmeansource, usepower, usec0, usecmn, usedelta,
                 useacc, n_last_frames, n_prev_frames, lofreq, hifreq,
                 mel_banks_only):
        self.audio_recorded_in = []

        self.ffnn = TheanoFFNN()
        self.ffnn.load(model)

        self.log_probs_speech = deque(maxlen=filter_length)
        self.log_probs_sil = deque(maxlen=filter_length)

        self.last_decision = 0.0


        self.front_end = MFCCFrontEnd(
            sample_rate, framesize,
            usehamming, preemcoef,
            numchans, ceplifter,
            numceps, enormalise,
            zmeansource, usepower,
            usec0, usecmn,
            usedelta, useacc,
            n_last_frames + n_prev_frames,
            lofreq, hifreq,
            mel_banks_only)

        self.framesize = framesize
        self.frameshift = frameshift

    def decide(self, data):
        """Processes the input frame whether the input segment is speech or non speech.

        The returned values can be in range from 0.0 to 1.0.
        It returns 1.0 for 100% speech segment and 0.0 for 100% non speech segment.
        """

        data = struct.unpack('%dh' % (len(data) / 2, ), data)
        self.audio_recorded_in.extend(data)

        while len(self.audio_recorded_in) > self.framesize:
            frame = self.audio_recorded_in[:self.framesize]
            self.audio_recorded_in = self.audio_recorded_in[self.frameshift:]

            mfcc = self.front_end.param(frame)

            prob_sil, prob_speech = self.ffnn.predict_normalise(mfcc.reshape(1,len(mfcc)))[0]

            # print prob_sil, prob_speech

            self.log_probs_speech.append(log(prob_speech))
            self.log_probs_sil.append(log(prob_sil))

            log_prob_speech_avg = 0.0
            for log_prob_speech, log_prob_sil in zip(self.log_probs_speech, self.log_probs_sil):
                log_prob_speech_avg += log_prob_speech - logsumexp([log_prob_speech, log_prob_sil])
            log_prob_speech_avg /= len(self.log_probs_speech)

            prob_speech_avg = np.exp(log_prob_speech_avg)

            # print 'prob_speech_avg: %5.3f' % prob_speech_avg

            self.last_decision = prob_speech_avg

        # returns a speech / non-speech decisions
        return self.last_decision