
import unittest
import coverage
import sys
import os
test_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(test_dir)
sys.path.append(package_dir)
COV_FLAG = True

if __name__ == "__main__":
    if COV_FLAG:
        cov = coverage.coverage()
        cov.start()

    loader = unittest.TestLoader()
    tests = loader.discover(test_dir, pattern='test_*.py')
    testRunner = unittest.runner.TextTestRunner()
    test_results = testRunner.run(tests)

    if COV_FLAG:
        cov.stop()
        cov.save()
        # 命令行模式展示结果
        cov.report()
        # 生成HTML覆盖率报告
        cov.html_report(directory=os.path.join(test_dir, 'covhtml'))
        cov.xml_report(outfile=os.path.join(test_dir, 'cov_report.xml'))


    if test_results.wasSuccessful():  # used in github actions to make sure actions fail when tests fails
        exit(0)
    else:
        exit(1)
