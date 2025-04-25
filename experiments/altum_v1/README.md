# Altum age

The goal of this experiment is to see whether we can create an agent team that is capable of achieving near-SOTA accuracy for predicting chronological age from DNA methylation. 

The pipeline will be composed of these five steps:

1. Understanding the task
2. Exploratory Data Analysis
3. Data splitting
4. Model training and evaluation
5. Discussion and iteration on experimental design
... Repeat until the model achieves near-SOTA accuracy

The data provided to the agents is the following:

- `betas.arrow`: a feather file containing the beta values for each sample. The rows are the sample IDs and the columns are the probes IDs.
- `metadata.arrow`: a feather file containing the metadata for each sample. The rows are the sample IDs and the columns are the metadata.

These are the agent personas:

- **Senior Advisor**: A well-respected late-career scientist with a deep understanding of the field. They ask questions to guide the research team. They provide valuable insights and feedback.
- **Principal Scientist**: A seasoned scientist who has a deep understanding of the task and the data. They are responsible for leading the research team, delegating tasks, making decisions, and writing meeting summaries and reports. They gain valuable insights from their own expertise, their exploration of the literature, and their interactions with the other agents.
- **Bioinformatics expert**: A bioinformatician who has a deep understanding of omics datasets. They also understand the best practices for preprocessing and analyzing these datasets. They have a high degree of expertise and hold themselves and others to a high standard when it comes to the best practices in bioinformatics.
- **Machine Learning expert**: A machine learning researcher who has a deep understanding of the latest research in the field. They provide valuable insights into the design of the model and the best approaches to use.
- **ML / data engineer**: An engineer who has a robust bioinformatics and machine learning background. Anything the other agents can plan, the engineer can deploy into code. The engineer will also have access to a code runner tool that can execute code and return the output. 

These are the tools available to the agents:

- **Perplexity**: A powerful search-based LLM that can be used to search the literature and answer both conceptual and technical questions.
- **Webpage Parser**: A tool that can parse webpages and extract the text.
- **Code runner**: A tool that can execute code and return the output. The environment is a conda environment with the following packages:
    - pandas
    - numpy
    - scikit-learn
    - scipy
    - matplotlib
    - seaborn
    - pytorch
- **Local file system**: A tool that can read and write files to the local file system.



### Understanding the task

In this step, our goal is to understand and decompose the task. The output of this step is a report from the principal scientist which answers the following questions:

- What is the purpose of this task? How does it relate to the overall goals of the research field in which the task is situated?
- What prior studies have been done on this task? What are the main approaches and methods used?
- What are the main challenges and limitations of the prior studies?
- What are the most promising approaches and methods for this task?

### Exploratory Data Analysis

Next, we need to generate some prompts for the next step (Exploratory Data Analysis). These prompts will be given to the principal scientist. The prompt will be the following:

```plaintext
The goal of the project is to predict chronological age from DNA methylation. Here is the specific task description: {task_description}

In the first meeting of the research team, we discussed the task and wrote a report summarizing the current state of the art in the field. Here is the report: {report}

Now, we need to begin the process of exploring the data.
```






