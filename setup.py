


from setuptools import setip, find_packages

packages = ["smdr"]

file_data = [
    ("smdr/data", ["smmdr/data/edges.csv", "smart/data/fmri_slice_zscores.csv"]),
]

requires = [
    "numpy", "scipy", "matplotlib", "pygfl"
]

#about = {}
#with open(os.path.join(here, 'smart', '__version__.py'), 'r', 'utf-8') as f:
#    exec(f.read(), about)

with open('README.rst', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name=about["smdr"],  
    version='0.0.1',  
    description=about["Spative Adaptive MDR Screening"],
    long_description=readme,   
    author=about["Yifei Hu","Xinge Jeng"],  
    author_email=about["yifei.hu0525@gmail.com"],  
    license="MIT",
    url=about["https://github.com/yifeihu93/smdr"],  
    packages=find_packages(),   
    data_files=file_data,   
    include_package_data=True, 
    python_requires=">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3*", 
    install_requires=requires,  
    zip_safe=False,  
    classifiers=[    
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)

