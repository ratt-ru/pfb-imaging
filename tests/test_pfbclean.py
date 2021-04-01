# from pfb.pfbclean import _main
# from pfb.parser.argparser import create_parser
# import packratt
# import os
# import traceback

# def test_pfb(tmp_path_factory):
#     test_dir = tmp_path_factory.mktemp("test_pfb")
#     packratt.get('/test/ms/2020-06-04/elwood/smallest_ms.tar.gz', str(test_dir))

#     # get default args
#     args = create_parser().parse_args([])
#     # populate required args
#     args.ms = [str(test_dir / 'smallest_ms.ms_p0')]
#     args.data_column = 'DATA'
#     args.weight_column = 'WEIGHT'
#     args.outfile = str(test_dir / 'image')
#     args.fov = 1.0
#     args.maxit = 2
#     args.minormaxit = 2
#     args.report_freq = 1
#     args.reweight_iters = 5
#     args.cgmaxit = 5
#     args.pmmaxit = 5
#     args.pdmaxit = 5

#     _main(args)