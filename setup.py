from distutils.core import setup
setup(
  name = 'classifygpt',         # How you named your package folder (MyLib)
  packages = ['classifygpt'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Classifying sentences as AI agent written or not',   # Give a short description about your library
  author = 'Aashai Avadhani',                   # Type in your name
  author_email = 'aashai123@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/AashaiAvadhani1/classifygpt',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
          'scikit-learn',
          'numpy',
          'pandas',
          'matplotlib'
      ],
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)