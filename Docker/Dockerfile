FROM choco1916/initials:cuda9-cudnn7-ubuntu16-conda3

USER jonghwan
WORKDIR /home/jonghwan
ENV HOME=/home/jonghwan
RUN chmod 777 /home/jonghwan

# Create a Python 3.6 environment
RUN conda install conda-build \
	&& conda create -y --name temporal_grounding python=3.6 \
	&& conda clean -ya
ENV CONDA_DEFAULT_ENV=temporal_grounding
ENV CONDA_PREFIX=/home/jonghwan/anaconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install python dependencies
RUN /bin/bash -c "source activate temporal_grounding"
RUN /bin/bash -c "python --version"
RUN conda install -n temporal_grounding -y nb_conda numpy \
	cython PyHamcrest scipy jupyter matplotlib \
	pillow nltk tqdm pyyaml seaborn scikit-image \
	scikit-learn h5py && conda clean -ya
RUN conda install -n temporal_grounding -y -c conda-forge \
	colorlog coloredlogs && conda clean -ya

# Install pytorch 1.1.0 with cuda 9.0
RUN conda install -n temporal_grounding pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch \ 
	&& conda clean -ya

# Install tensorflow for using tensorboard
RUN pip install --upgrade pip
RUN pip install tensorflow

# Set the default command to run temporal_grounding envrionment
CMD ["source activate temporal_grounding"]
