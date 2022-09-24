
from utils     import die
from server    import app
from argparse  import ArgumentParser, Namespace
from inference import classify

import time


def main():
    args = parse_cmdline_args()
    match args:

        case Namespace(server=True, link=_):
            app.run()

        case Namespace(server=_, link=website):
            if website is None:
                die('error: either pass in a link or activate the server')

            else:
                start = time.perf_counter()
                classified = classify(website)
                end = time.perf_counter()

                if classified is None:
                    die('cannot classify website')

                prediction, probability = classified

                print(f'       {website}\ncategory: {prediction}\ntime elapsed: {end - start}\n')


def parse_cmdline_args() -> Namespace:
    '''
    parse command line arguments
    '''
    cmdline = ArgumentParser(description='website classifier')

    options = cmdline.add_mutually_exclusive_group()
    '''
    start the server
        main.py --server
    '''
    options.add_argument( '--server', dest = 'server', action = 'store_true', default = False
                        , help = 'start the web server' )

    '''
    pass in a link to classify
        main.py --link 'https://hoogle.haskell.org'
    '''
    options.add_argument( '--link' , dest = 'link' , type = str
                        , help = 'link to website' )

    return cmdline.parse_args()


if __name__ == '__main__':
    main()

