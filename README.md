
<a id="readme-top"></a>


<!-- PROJECT LOGO -->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project, developed during an internship at SafeQual, leverages OpenAI's API to enhance incident reports through semantic search and inference. It includes Python scripts that perform semantic searches, generate context-gathering questions, and augment reports with user-provided answers, utilizing machine learning techniques such as Euclidean distance ranking, K-means clustering, vector similarity, embeddings, text generation, and retrieval augmented generation (RAG). The system features automated and interactive workflows designed to iteratively improve and enrich medical incident reports, ensuring accurate and context-rich responses.

Here's why:
* **Enhancing Accuracy:** Ensures accurate and detailed incident reports, reducing misinterpretation and missing information.
* **Supporting Providers:** Helps healthcare professionals make informed decisions, improving patient care and outcomes.
* **Educating Patients:** Provides patients with accurate medical information, empowering informed healthcare decisions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python-Badge]][Python-url]
* [![OpenAI-Badge]][OpenAI-url]
* [![SciPy-Badge]][SciPy-url]
* [![scikit-learn-Badge]][scikit-learn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

First, make sure to have Python installed on your system. This project was implemented using Python version 3.11, so that version 3.11 or newer should suffice. [https://www.python.org/downloads/](https://www.python.org/downloads/)

### Installation

_Similar instructions for setting up Python and the OpenAI environment can be found on OpenAI's quickstart tutorial [https://www.python.org/downloads/](https://www.python.org/downloads/), however there are also additional Python packages that must be installed (below)._

1. Create or login to your OpenAI account [https://platform.openai.com/](https://platform.openai.com/)
2. Create an OpenAI API key [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Save the value of the API key as an environment variable under the variable name "OPENAI_API_KEY"
_Here is a concise tutorial on setting this up: [https://www.immersivelimit.com/tutorials/adding-your-openai-api-key-to-system-environment-variables](https://www.immersivelimit.com/tutorials/adding-your-openai-api-key-to-system-environment-variables)_
4. Create an empty project folder and clone the repo
   ```sh
   git clone https://github.com/Alex2539x/safequal-report-enhancement.git
   ```
5. Set up virtual environment (recommended)
   ```sh
   python -m venv openai-env
   ```
   For windows:
   ```sh
   openai-env\Scripts\activate
   ```
   Or for Unix or MacOS, run:
   ```sh
   source openai-env/bin/activate
   ```
6. Install OpenAI package
   ```sh
   pip install --upgrade openai
   ```
7. Install other related Python packages
   ```sh
   pip install matplotlib
   ```
   ```sh
   pip install scikit-learn
   ```   
   ```sh
   pip install scipy
   ```   

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

To run the scripts, specific data json files are needed in order to generate an output. DUe to this, the recommended order of script calls is as follows (chronologically):

generate.py -> embeddings.py -> k-means-silhouette.py -> k-means.py -> search-and-inference.py

The other unlisted scripts are for your own personal use to analyze the existing json data, evaluate accuracy of categorized clusters, etc. Additionally, certain scripts act as helper files for other primary scripts.  

In order to run any of the scripts in this project, run the following command (template):
   ```sh
   python <script-name> <applicable-arguments>
   ```   
Examples:
   ```sh
   python generate.py "No topic description provided" "medication" 5 
   ``` 
   ```sh
   python k-means.py embeddings.json 11
   ``` 
Additional examples for each script are listed at the bottoms of each script file.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Alex Ho - [https://alextho.com/](https://alextho.com/) - alex2539x@gmail.com

Project Link: [https://github.com/Alex2539x/safequal-report-enhancement](https://github.com/Alex2539x/safequal-report-enhancement)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python-Badge]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge
[Python-url]: https://www.python.org/downloads/

[OpenAI-Badge]: https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=fff&style=for-the-badge
[OpenAI-url]: https://platform.openai.com/

[SciPy-Badge]: https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=fff&style=for-the-badge
[SciPy-url]: https://scipy.org/

[scikit-learn-Badge]: https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=fff&style=for-the-badge
[scikit-learn-url]: https://scikit-learn.org/stable/