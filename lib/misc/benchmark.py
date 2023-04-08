#!usr/bin/env python

# -----------------------------------------------------------------------------
"""
benchmark.py: Module to set up an simulation benchmark for tests and
validations
"""

__author__      = "Fangda Li"
__copyright__   = "Copyright (C) 2020, Robot Vision Lab"
__date__        = "25th December, 2020"
__credits__     = ["Ankit Manerikar", "Fangda Li", "Dr. Avinash Kak"]
__license__     = "Public Domain"
__version__     = "2.0.0"
__maintainer__  = ["Ankit Manerikar", "Fangda Li"]
__email__       = ["amanerik@purdue.edu", "li1208@purdue.edu"]
__status__      = "Prototype"
# ------------------------------------------------------------------------------


import time, os, sys
from lib.misc.util import Logger


class Benchmark(object):

    def __init__(self, save_log=True, save_remark=True):
        """
        ------------------------------------------------------------------------
         Constructor.

        :param save_log:    flag to enable logging of results
        :param save_remark: flag to save remark for the benchmark
        ------------------------------------------------------------------------
        """

        self.remark = "Replace me with your remark on this benchmark!"
        self.i = 0
        self.t_str = ''
        self.t0 = time.time()
        self.save_log = save_log
        self.save_remark = save_remark
        self.out_dir_list = None
    # --------------------------------------------------------------------------

    def set_remark(self, remark):
        """
        ------------------------------------------------------------------------
        Sets the benchmark remark

        :param remark:  remark string
        :return:
        ------------------------------------------------------------------------
        """
        self.remark = remark
    # --------------------------------------------------------------------------

    def set_test_cases(self, test_cases):
        """
        ------------------------------------------------------------------------
        Set the input directory from the to load data

        :param test_cases: List of test cases to be included in the benchmark
        :return:
        ------------------------------------------------------------------------

        """

        self.in_file_list = test_cases
    # --------------------------------------------------------------------------

    def set_output_dir(self, out_dir_list):
        """
        ------------------------------------------------------------------------
        Set the output directory for saving result data

        :param out_dir_list: list of directory names for each instance in input
        :return:
        ------------------------------------------------------------------------
        """

        if out_dir_list is None:
            self.out_dir_list = None
        else:
            for d in out_dir_list:
                os.makedirs(d, exist_ok=True)
            self.out_dir_list = out_dir_list
    # --------------------------------------------------------------------------

    def set_handles(self, preprocess, run, postprocess, alldone=None):
        """
        ------------------------------------------------------------------------
        Set function handles for the benchmark. See Example for understanding
        usage.

        :param preprocess:  function handle for preprocessing code
        :param run:         function handle for test code
        :param postprocess: function handle for postprocessing code
        :param alldone:     function handle for exiting code
        :return:
        ------------------------------------------------------------------------
        """

        # Function handles to be overloaded
        self.preprocess = preprocess
        self.run = run
        self.postprocess = postprocess
        self.alldone = alldone
    # --------------------------------------------------------------------------

    def start(self):
        """
        ------------------------------------------------------------------------
        Run the benchmark with all the function handles set

        :return:
        ------------------------------------------------------------------------
        """

        if self.out_dir_list == None:
            self.out_dir_list = [''] * len(self.in_file_list)
        for in_file, out_dir in zip(self.in_file_list, self.out_dir_list):
            pre = self._preprocess(in_file)
            res = self._run(pre, out_dir)
            self._postprocess(res, out_dir)
        return self._alldone()
    # --------------------------------------------------------------------------

    def _preprocess(self, path):
        """
        ------------------------------------------------------------------------
        Do necessary preprocessing before launching the test. E.g.
        instantiating a reconstructor.

        :param path:    input path
        :return:
        ------------------------------------------------------------------------
        """

        pre = self.preprocess(path)
        return pre
    # --------------------------------------------------------------------------

    def _run(self, pre, out_dir):
        """
        ------------------------------------------------------------------------
        Run the test and return the results

        :param pre:     data passed from preprocessing directory
        :param out_dir: output directory
        :return:
        ------------------------------------------------------------------------
        """

        if out_dir is not None:
            if self.save_log:
                self.t_str = time.strftime("DEBISIM-DATA-LOG-%Y%b%d-%H%M")
                logname = os.path.join(out_dir, self.t_str + '.txt')
                sys.stdout = Logger(logname)
            if self.save_remark:
                with open(os.path.join(out_dir, 'README.txt'), 'w') as f:
                    f.write(self.remark)

        print('BENCHMARK: Start, saving results to %s' % out_dir)
        print('BENCHMARK: %s' % self.remark)
        tik = time.time()
        res = self.run(pre)
        tok = time.time()
        print('BENCHMARK: Done, took %.5fs' % (tok - tik))
        if out_dir is not None and self.save_log:
            sys.stdout = sys.__stdout__
        return res
    # --------------------------------------------------------------------------

    def _postprocess(self, res, out_dir):
        """
        ------------------------------------------------------------------------
        Do necessary postprocessing, e.g. saving the results.

        :param res:     Result data passed from test code
        :return:
        ------------------------------------------------------------------------
        """

        post = self.postprocess(res, out_dir)
        self.i += 1
    # --------------------------------------------------------------------------

    def _alldone(self):
        """
        ------------------------------------------------------------------------
        Exiting code - any data to be return to parent code.

        :return:
        ------------------------------------------------------------------------
        """
        print('BENCHMARK: All done!! Took %.5fs' % (time.time() - self.t0))

        if self.alldone is not None:
            return self.alldone()
        else:
            return None
    # --------------------------------------------------------------------------

