# CPU runtime. Versions match the archived environment; TF uses its CPU wheel.
FROM python:3.7.12-bullseye@sha256:b48983bebd0fe1c09639fa008e4cb51aac6277af6c6762fc58ac9d2cb7fc24ef AS runtime
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    RESS_FINDER_THREADS=1 CUDA_VISIBLE_DEVICES=-1 MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /work
# This pinned buildpack-deps image already supplies the compiler and Cairo headers.
RUN command -v g++ && command -v pkg-config && pkg-config --modversion cairo
COPY environment/requirements-cpu.txt /tmp/requirements.txt
RUN python -m pip install --index-url "$PIP_INDEX_URL" --no-cache-dir pip==22.1.2 setuptools==65.6.3 wheel==0.34.2 && \
    python -m pip install --index-url "$PIP_INDEX_URL" --no-cache-dir --no-deps -r /tmp/requirements.txt
COPY run/vendor/FINDER_ND /work/run/vendor/FINDER_ND
RUN cd run/vendor/FINDER_ND && python setup.py build_ext --inplace --force

FROM runtime AS publication
COPY run /work/run
COPY real_data /work/real_data
COPY reproduction /work/reproduction
COPY results/R_mean.csv results/appendix_c_statistics.zip /work/results/
COPY reproduce.py /work/reproduce.py
CMD ["python", "reproduce.py"]
