from scripts.pfbclean import main, create_parser
import packratt
import os
import traceback

def test_pfb(test_dir):
    # get default args
    args = create_parser().parse_args()
    # populate required args
    args.ms = [test_dir + 'smallest_ms.ms_p0']
    args.data_column = 'DATA'
    args.weight_column = 'WEIGHT'
    args.outfile = test_dir + 'image'
    args.fov = 1.0
    args.maxit = 2
    args.report_freq = 1
    args.reweight_iters = 1
    args.cgmaxit = 5
    args.pmmaxit = 5
    args.pdmaxit = 5

    main(args)

    

if __name__=="__main__":
    # get data
    test_dir = 'scripts/test/tmp/'
    os.system('mkdir %s'%test_dir)
    packratt.get('/test/ms/2020-06-04/elwood/smallest_ms.tar.gz', test_dir)

    try:
        test_pfb(test_dir)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)
    finally:
        os.system('rm -r %s'%test_dir)

    