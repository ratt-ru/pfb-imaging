import packratt
import os
import traceback

def test_pfb(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("test_pfb")
    packratt.get('/test/ms/2020-06-04/elwood/smallest_ms.tar.gz', str(test_dir))

    os.system('pfbclean --ms {0} --data_column DATA --weight_column WEIGHT '
              '--outfile {1} --fov 1.0 --maxit 2 --minormaxit 2 '
              '--report_freq 1 --reweight_iters 5 --cgmaxit 5 --cgminit 5 '
              '--pmmaxit 5 --pdmaxit 5'.format([str(test_dir / 'smallest_ms.ms_p0')],
              str(test_dir / 'image')))