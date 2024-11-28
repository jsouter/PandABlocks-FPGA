#!/usr/bin/env python
import argparse
import configparser
import os
import logging
import shutil
import time
import subprocess
import csv
import pandas as pd

from pathlib import Path
from typing import Dict, List

import cocotb
import cocotb.handle

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly
from cocotb_tools import runner

from dma_driver import DMADriver

logger = logging.getLogger(__name__)

SCRIPT_DIR_PATH = Path(__file__).parent.resolve()
TOP_PATH = SCRIPT_DIR_PATH.parent.parent
MODULES_PATH = TOP_PATH / 'modules'
WORKING_DIR = Path.cwd()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    parser.add_argument('test_name', nargs='?', default=None)
    parser.add_argument('--sim', default='nvc')
    parser.add_argument('--skip', default=None)
    parser.add_argument('--panda-build-dir', default='/build')
    parser.add_argument('-c', action='store_true')
    return parser.parse_args()

def read_ini(path: List[str] | str) -> configparser.ConfigParser:
    """Read INI file and return its contents.

    Args:
        path: Path to INI file.
    Returns:
        ConfigParser object containing INI file.
    """
    app_ini = configparser.ConfigParser()
    app_ini.read(path)
    return app_ini


def get_timing_ini(module):
    """Get a module's timing INI file.

    Args:
        module: Name of module.
    Returns:
        Contents of timing INI.
    """
    ini_path = (MODULES_PATH / module / '{}.timing.ini'.format(module))
    return read_ini(str(ini_path.resolve()))

def block_has_dma(block_ini):
    """Check if module requires a dma to work.

    Args:
        block_ini: INI file containing signals information about a module.
    """
    return block_ini['.'].get('type', '') == 'dma'


def block_is_pcap(block_ini):
    """Check if module is pcap.

    Args:
        block_ini: INI file containing signals information about a module.
    """
    return block_ini['.'].get('type') == 'pcap'


def get_module_build_args(module, panda_build_dir):
    """Get simulation build arguments from a module's test config file.

    Args:
        module: Name of module.
    Returns:
        List of extra build arguments.
    """
    test_config_path = MODULES_PATH / module / 'test_config.py'
    if test_config_path.exists():
        g = {'TOP_PATH': TOP_PATH,
            'BUILD_PATH': Path(panda_build_dir)}
        code = open(str(test_config_path)).read()
        exec(code, g)
        args = g.get('EXTRA_BUILD_ARGS', [])
        return args
    return []


def order_hdl_files(hdl_files, build_dir, top_level):
    """Put vhdl source files in compilation order. This is neccessary for the
    nvc simulator as it does not order the files iself before compilation.

    Args:
        hdl_files: List of vhdl source files.
        build_dir: Build directory for simulation.
        top_level: Name of the top-level entity.
    """
    command = ['vhdeps', 'dump', top_level, '-o',
               f'{WORKING_DIR / build_dir / "order"}']
    for file in hdl_files:
        command.append(f'--include={str(file)}')
    command_str = ' '.join(command)
    Path(WORKING_DIR / build_dir).mkdir(exist_ok=True)
    subprocess.run(['/usr/bin/env'] + command)
    try:
        with open(TOP_PATH / build_dir / 'order') as order:
            ordered_hdl_files = \
                [line.strip().split(' ')[-1] for line in order.readlines()]
        return ordered_hdl_files
    except FileNotFoundError as error:
        logger.warning(f'Likely that the following command failed:\n{command_str}')
        logger.warning(error)
        logger.warning('HDL FILES HAVE NOT BEEN PUT INTO COMPILATION ORDER!')
        return hdl_files


def get_module_hdl_files(module, top_level, build_dir, panda_build_dir):
    """Get HDL files needed to simulate a module from its test config file.

    Args:
        module: Name of module.
    Returns:
        List of paths to the HDL files.
    """
    module_dir_path = MODULES_PATH / module
    test_config_path = module_dir_path / 'test_config.py'
    g = {'TOP_PATH': TOP_PATH,
         'BUILD_PATH': Path(panda_build_dir)}
    if test_config_path.exists():
        code = open(str(test_config_path)).read()
        exec(code, g)
        g.get('EXTRA_HDL_FILES', [])
        extra_files = list(g.get('EXTRA_HDL_FILES', []))
        extra_files_2 = []
        for my_file in extra_files:
            if str(my_file).endswith('.vhd'):
                extra_files_2.append(my_file)
            else:
                extra_files_2 = extra_files_2 + list(my_file.glob('**/*.vhd'))
    else:
        extra_files_2 = []
    result = extra_files_2 + list((module_dir_path / 'hdl').glob('*.vhd'))
    result = order_hdl_files(result, build_dir, top_level)
    logger.info('Gathering the following VHDL files:')
    for my_file in result:
        logger.info(my_file)
    return result


def get_module_top_level(module, panda_build_dir):
    """Get top level entity from a module's test config file.
    If none is found, assume top level entity is the same as the module name.

    Args:
        module: Name of module.
    Returns:
        Name of top level entity.
    """
    test_config_path = MODULES_PATH / module / 'test_config.py'
    if test_config_path.exists():
        g = {'TOP_PATH': TOP_PATH,
            'BUILD_PATH': Path(panda_build_dir)}
        code = open(str(test_config_path)).read()
        exec(code, g)
        top_level = g.get('TOP_LEVEL', None)
        if top_level:
            return top_level
    return module


def print_results(module, passed, failed, time=None):
    """Format and print results from a module's tests.

    Args:
        module: Name of module.
        passed: List of the names of tests that passed.
        failed: List of the names of tests that failed.
        time: Time taken to run the tests.
    """
    print('__')
    print('\nModule: {}'.format(module))
    if len(passed) + len(failed) == 0:
        print('\033[0;33m' + 'No tests ran.' + '\033[0m')
    else:
        percentage = round(len(passed) / (len(passed) + len(failed)) * 100)
        print('{}/{} tests passed ({}%).'.format(
            len(passed), len(passed) + len(failed), percentage))
        if time:
            print('Time taken = {}s.'.format(time))
        if failed:
            print('\033[0;31m' + 'Failed tests:' + '\x1b[0m', end=' ')
            print(*[test + (', ' if i < len(failed) - 1 else '.')
                    for i, test in enumerate(failed)])
        else:
            print('\033[92m' + 'ALL PASSED' + '\x1b[0m')


def summarise_results(results):
    """Format and print summary of results from a test run.

    Args:
        Results: Dictionary of all results from a test run.
    """
    failed = [module for module in results if results[module][1]]
    passed = [module for module in results if not results[module][1]]
    total_passed, total_failed = 0, 0
    for module in results:
        total_passed += len(results[module][0])
        total_failed += len(results[module][1])
    total = total_passed + total_failed
    print('\nSummary:\n')
    if total == 0:
        print('\033[1;33m' + 'No tests ran.' + '\033[0m')
    else:
        print('{}/{} modules passed ({}%).'.format(
            len(passed), len(results.keys()),
            round(len(passed) / len(results.keys()) * 100)))
        print('{}/{} tests passed ({}%).'.format(
            total_passed, total, round(total_passed / total * 100)))
        if failed:
            print('\033[0;31m' + '\033[1m' + 'Failed modules:' +
                  '\x1b[0m', end=' ')
            print(*[module + (', ' if i < len(failed) - 1 else '.')
                    for i, module in enumerate(failed)])
        else:
            print('\033[92m' + '\033[1m' + 'ALL MODULES PASSED' + '\x1b[0m')


def get_simulator_build_args(simulator):
    if simulator == 'ghdl':
        return ['--std=08', '-fsynopsys', '-Wno-hide']
    elif simulator == 'nvc':
        return ['--std=2008']


def get_test_args(simulator, build_args, test_name):
    test_name = test_name.replace(' ', '_').replace('/', '_')
    if simulator == 'ghdl':
        return build_args
    elif simulator == 'nvc':
        return ['--ieee-warnings=off', f'--wave={test_name}.vcd']


def get_elab_args(simulator):
    if simulator == 'nvc':
        return ['--cover']
    else:
        return []


def get_plusargs(simulator, test_name):
    test_name = test_name.replace(' ', '_').replace('/', '_')
    vcd_filename = f'{test_name}.vcd'
    if simulator == 'ghdl':
        return [f'--vcd={vcd_filename}']
    elif simulator == 'vcd':
        return []
    return []


def collect_coverage_file(build_dir, top_level, test_name):
    coverage_path = Path(WORKING_DIR / build_dir / 'coverage')
    Path(coverage_path).mkdir(exist_ok=True)
    old_file_path = Path(WORKING_DIR / build_dir / 'top' /
                         f'_TOP.{top_level.upper()}.elab.covdb')
    test_name = test_name.replace(" ", "_").replace("/", "_")
    new_file_path = Path(coverage_path /
                         f'_TOP.{top_level.upper()}.{test_name}.elab.covdb')
    subprocess.run(['mv', old_file_path, new_file_path])
    return new_file_path


def merge_coverage_data(build_dir, module, file_paths):
    merged_path = Path(WORKING_DIR / build_dir / 'coverage' /
                       f'merged.{module}.covdb')
    command = ['nvc', '--cover-merge', '-o'] + \
              [str(merged_path)] + \
              [str(file_path) for file_path in file_paths]
    subprocess.run(command)
    return merged_path


def cleanup_dir(test_name, build_dir):
    test_name = test_name.replace(' ', '_').replace('/', '_')
    (WORKING_DIR / build_dir / test_name).mkdir()
    logger.info(f'Putting all files related to "{test_name}" in {str(
        WORKING_DIR / build_dir / test_name)}')
    for file in (WORKING_DIR / build_dir).glob(f'{test_name}*'):
        if file.is_file():
            new_name = str(file).split('/')[-1].replace(test_name, '')
            new_name = new_name.lstrip('_')
            file.rename(WORKING_DIR / build_dir / test_name / new_name)


def print_errors(failed_tests, build_dir):
    for test_name in failed_tests:
        logger.info(f'        See timing errors for "{test_name}" below')
        test_name = test_name.replace(' ', '_').replace('/', '_')
        with open(WORKING_DIR / build_dir / test_name / 'errors.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                logger.timing_error(row[1])


def print_coverage_data(coverage_report_path):
    print('Code coverage:')
    coverage_path = coverage_report_path.parent
    command = ['nvc', '--cover-report', '-o', str(coverage_path),
               str(coverage_report_path)]
    subprocess.run(command)


def setup_logger():
    timing_error_level = 30
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s: %(message)s")
    logging.addLevelName(30, "TIMING_ERROR")
    def timing_error(self, message, *args, **kwargs):
        if self.isEnabledFor(timing_error_level):
            self._log(timing_error_level, message, args, **kwargs)
    logging.Logger.timing_error = timing_error


def test_module(module, test_name=None, simulator='nvc',
                panda_build_dir='/build', collect=False):
    """Run tests for a module.

    Args:
        module: Name of module.
        test_name: Name of specific test to run. If not specified, all tests
            for that module will be run.
    Returns:
        Lists of tests that passed and failed respectively, path to coverage.
    """
    timing_ini = get_timing_ini(module)
    if not Path(MODULES_PATH / module).is_dir():
        raise FileNotFoundError('No such directory: \'{}\''.format(
            Path(MODULES_PATH / module)))
    if test_name:
        sections = []
        for test in test_name.split(',,'):   # Some test names contain a comma
            if test in timing_ini.sections():
                sections.append(test)
            else:
                print('No test called "{}" in {} INI timing file.'
                    .format(test, module)
                    .center(shutil.get_terminal_size().columns))
        if not sections:
            return [], [], None
    else:
        sections = timing_ini.sections()
    sim = runner.get_runner(simulator)
    build_dir = f'sim_build_{module}'
    build_args = get_simulator_build_args(simulator)
    build_args += get_module_build_args(module, panda_build_dir)
    top_level = get_module_top_level(module, panda_build_dir)
    sim.build(sources=get_module_hdl_files(module, top_level, build_dir,
                                           panda_build_dir),
              build_dir=build_dir,
              hdl_toplevel=top_level,
              build_args=build_args,
              clean=True)
    passed, failed = [], []
    coverage_file_paths = []
    coverage_report_path = None

    for section in sections:
        if section.strip() != '.':
            test_name = section
            print()
            print('Test: "{}" in module {}.\n'.format(test_name, module))
            xml_path = sim.test(hdl_toplevel=top_level,
                                test_module='cocotb_simulate_test',
                                build_dir=build_dir,
                                test_args=get_test_args(simulator, build_args,
                                                        test_name),
                                elab_args=get_elab_args(simulator),
                                plusargs=get_plusargs(simulator, test_name),
                                extra_env={'module': module,
                                           'test_name': test_name,
                                           'simulator': simulator,
                                           'sim_build_dir': build_dir,
                                           'panda_build_dir': panda_build_dir,
                                           'collect': collect})
            results = runner.get_results(xml_path)
            if simulator == 'nvc':
                coverage_file_paths.append(
                    collect_coverage_file(build_dir, top_level, test_name))
            if results == (1, 0):
                # ran 1 test, 0 failed
                passed.append(test_name)
            elif results == (1, 1):
                # ran 1 test, 1 failed
                failed.append(test_name)
            else:
                raise ValueError(f'Results unclear: {results}')
            cleanup_dir(test_name, build_dir)
    if simulator == 'nvc':
        coverage_report_path = merge_coverage_data(
            build_dir, module, coverage_file_paths)
    return passed, failed, coverage_report_path


def get_cocotb_testable_modules():
    """Get list of modules that contain a test config file.

    Returns:
        List of names of testable modules.
    """
    modules = MODULES_PATH.glob('*/*.timing.ini')
    return list(module.parent.name for module in modules)


def run_tests():
    """Perform test run.
    """
    t_time_0 = time.time()
    args = get_args()
    setup_logger()
    if args.module.lower() == 'all':
        modules = get_cocotb_testable_modules()
    else:
        modules = args.module.split(',')
    skip_list = args.skip.split(',') if args.skip else []
    for module in skip_list:
        if module in modules:
            modules.remove(module)
            print(f'Skipping {module}.')
        else:
            print(f'Cannot skip {module} as it was not going to be tested.')
    simulator = args.sim
    collect = 'True' if args.c else 'False'
    results = {}
    times = {}
    coverage_reports = {}
    for module in modules:
        t0 = time.time()
        module = module.strip('\n')
        build_dir = f'sim_build_{module}'
        results[module] = [[], []]
        # [[passed], [failed]]
        print()
        print('* Testing module \033[1m{}\033[0m *'.format(module.strip("\n"))
              .center(shutil.get_terminal_size().columns))
        print('---------------------------------------------------'
              .center(shutil.get_terminal_size().columns))
        results[module][0], results[module][1], coverage_reports[module] = \
            test_module(module, test_name=args.test_name, simulator=simulator,
                        panda_build_dir=args.panda_build_dir, collect=collect)
        t1 = time.time()
        times[module] = round(t1 - t0, 2)
    print('___________________________________________________')
    print('\nResults:')
    for module in results:
        print_results(module, results[module][0], results[module][1],
                      times[module])
        if coverage_reports[module] is not None:
            print_coverage_data(coverage_reports[module])
        print_errors(results[module][1], build_dir)
    print('___________________________________________________')
    summarise_results(results)
    t_time_1 = time.time()
    print('\nTime taken: {}s.'.format(round(t_time_1 - t_time_0, 2)))
    print('___________________________________________________\n')
    print(f'Simulator: {simulator}\n')


def get_ip(module=None, quiet=True):
    if not module:
        modules = os.listdir(MODULES_PATH)
    else:
        modules = [module]
    ip = {}
    for module in modules:
        ini = get_block_ini(module)
        if not ini.sections():
            if not quiet:
                print('\033[1m' + f'No block INI file found in {module}!'
                      + '\033[0m')
            continue
        info = []
        if '.' in ini.keys():
            info = ini['.']
        spaces = ' ' + '-' * (16 - len(module)) + ' '
        if 'ip' in info:
            ip[module] = info['ip']
            if not quiet:
                print('IP needed for module \033[1m' + module + '\033[0m:' +
                      spaces + '\033[0;33m' + info['ip'] + '\033[0m')
        else:
            ip[module] = None
            if not quiet:
                print('IP needed for module ' + '\033[1m' + module
                      + '\033[0m:' + spaces + 'None found')
    return ip


def check_timing_ini(module=None, quiet=True):
    if not module:
        modules = os.listdir(MODULES_PATH)
    else:
        modules = [module]
    has_timing_ini = {}
    for module in modules:
        ini = get_timing_ini(module)
        if not ini.sections():
            has_timing_ini[module] = False
            if not quiet:
                print('\033[0;33m' +
                      f'No timing INI file found in - \033[1m{module}\033[0m')
        else:
            has_timing_ini[module] = True
            if not quiet:
                print(f'Timing ini file found in ---- \033[1m{module}\033[0m')
    return has_timing_ini


def get_some_info():
    ini_and_ip = []
    ini_no_ip = []
    no_ini = []
    has_timing_ini = check_timing_ini()
    has_ip = get_ip()
    for module in has_timing_ini.keys():
        if has_timing_ini[module]:
            if has_ip[module]:
                ini_and_ip.append(module)
            else:
                ini_no_ip.append(module)
        else:
            no_ini.append(module)
    print('\nModules with no timing INI:')
    for i, module in enumerate(no_ini):
        print(f'{i + 1}. {module}')
    print('\nModules with timing INI and IP:')
    for i, module in enumerate(ini_and_ip):
        print(f'{i + 1}. {module}')
    print('\nModules with timing INI and no IP:')
    for i, module in enumerate(ini_no_ip):
        print(f'{i + 1}. {module}')


def main():
    args = get_args()
    if args.module.lower() == 'ip':
        get_ip(args.test_name, quiet=False)
    elif args.module.lower() == 'ini':
        check_timing_ini(quiet=False)
    elif args.module.lower() == 'info':
        get_some_info()
    else:
        run_tests()


if __name__ == "__main__":
    main()
