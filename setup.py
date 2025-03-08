from setuptools import find_packages,setup
from typing import List



HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
  """This function returns a list of requirements from a given file path."""
  requirements =[]
  with open(file_path) as f:
    requirements = f.read().readlines()
    requirements=[req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
      requirements.remove(HYPEN_E_DOT)
  return requirements




setup(
  name='mlproject',
  version ='0.0.1',
  author ='Ashis',
  author_email ='ashis@ashis.com',
  packages = find_packages(),
  install_requires=get_requirements('requirements.txt')
  # install_requires=['numpy','pandas','matplotlib','scikit-learn'],

)