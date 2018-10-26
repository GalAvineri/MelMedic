import os
import shutil


class IO:
    @staticmethod
    def create_or_recreate_dir(dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    @staticmethod
    def create_if_none(dir_path):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def validate_exists_and_dir(dir_path, arg_name):
        if not os.path.exists(dir_path):
            raise ValueError("{0} {1} does not exist".format(arg_name, dir_path))

        if not os.path.isdir(dir_path):
            raise ValueError("{0} {1} is not a dir".format(arg_name, dir_path))


class MultiProcessing:
    @staticmethod
    def imap_wrapper(args):
        """
        :param args: tuple of the form (func, f_arguments)
        :return: result of func(**f_arguments)
        """

        func = args[0]
        f_args = args[1:]
        return func(*f_args)


class Reproduction:
    @staticmethod
    def make_run_reproducible():
        from numpy.random import seed
        from tensorflow import set_random_seed
        seed(1)
        set_random_seed(2)
