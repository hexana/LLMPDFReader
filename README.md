# LLMPDFReader
how to use llm to read the pdf files. Attached doc explains Executing LLM model using LangChain and hugging face.




With all the hype of Generative AI, I want to try how to use the LLM model for a simple application. I built a program to read pdf files using models available on the hugging face site (https://huggingface.co/) . There are multiple articles and videos available for this work and I am not going to do something new, but I am writing this mostly because of problems I faced during the setup. Maybe this will help someone who is also trying a similar setup.
 I used the Nvidia L4 system to run this code.

1.	Create an account in hugging face site:
 ![huggingFace_signup](https://github.com/user-attachments/assets/1cde3544-e452-46a6-b44c-941027a5357e)



2.	Create the API token.
 ![create_token](https://github.com/user-attachments/assets/17dd2d99-38c9-4a15-8119-805dc218b8d7)


3.	search the model you wanted to use, in my case, I used mistralai/Mistral-7B-v0.3. select the agreement for the model.
 ![model_permission](https://github.com/user-attachments/assets/7e3c3c22-52c2-4b77-850e-d3b3d1204f0b)


Go to the token→edit permission  and provide access.
 ![model_permission](https://github.com/user-attachments/assets/5417bbd0-23d3-447c-8af0-3e149cb128b7)


4.	Install python >=3.9 if you don’t have one already. 
5.	Install hugging face hub and cli using pip - pip install -U "huggingface_hub[cli]"
6.	Install below packages:
Pip install transformers
Pip install protobuf
Pip install sentencepiece
pip install langchain-community

7.	Now the problem I faced was to download the model and execute. I used the code which is available at model page in huggingface site:
   
from huggingface_hub import snapshot_download
from pathlib import Path
mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

First you need to login to hugginface cli then only this download works and later the problem will occur. Once you try to use model (with the correct path where you installed the model) the program  will always throw
Can't load tokenizer If you were trying to load it from ‘Models - Hugging Face’, make sure you don’t have a local directory with the same name.

how to login to huggingface cli? In one of the setup, I was able to without any issues with command huggingface-cli login. In another setup (on mac), I had to run the command echo "export PATH=\"`python3 -m site --user-base`/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
and then called  huggingface-cli login.This will ask huggingface api token.
8.	On the same terminal, where you logged in to  huggingface-cli,  download the model and tokenizer. Use the code I have attached in git LLMDownloader.py, change the path where you want to save the model and tokenizer. Run using 
Python LLMDownloader.py  
The path will be used in another file to use the downloaded model. Notice we can change the default path for hugging face home, which is under ./cache for model, dataset etc.

9.	Now use the code PDFQA.py to get answers from pdf.  Update the code for the location of pdf files as well as for the model and tokenizer file. You can notice how the tokenizer and model are used in this file. I am not going into details of each method/code as there is already plenty of material available online. 


Hope you will be able to execute a LLM model for your project.



	



 

