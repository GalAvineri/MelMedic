def imap_wrapper(args):
    """
    :param args: tuple of the form (func, f_arguments)
    :return: result of func(**f_arguments)
    """

    func = args[0]
    f_args = args[1:]
    return func(*f_args)