from setuptools import setup, find_packages

setup(
    name='dmgp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'torch>=1.9.0',
        'scipy>=1.12.0',
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # 'your_command=your_package.module:function',
        ],
    },
    author='Wenyuan Zhao, Haoyuan Chen',
    author_email='wyzhao@tamu.edu',
    description='A sparse expansion for deep Gaussian processes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/warrenzha/dmgp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)


if __name__ == "__main__":
    setup()
