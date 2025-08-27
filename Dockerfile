FROM python:3.11-slim

# Envs propres + config Streamlit dans /tmp (writable)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    # caches/libs écrivent dans /tmp
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp \
    # <- chemin de config Streamlit (évite /app/.streamlit non writable)
    STREAMLIT_CONFIG_DIR=/tmp/.streamlit \
    # <- orthographe correcte (avec underscores)
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # qualité de vie
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Code
COPY . /app

# Prépare /tmp/.streamlit et /tmp/matplotlib
# - copie le config.toml du repo s'il existe
# - sinon crée un fichier minimal
# - dans tous les cas force gatherUsageStats=false
RUN set -eux; \
    mkdir -p /tmp/.streamlit /tmp/matplotlib; \
    if [ -f /app/.streamlit/config.toml ]; then \
        cp /app/.streamlit/config.toml /tmp/.streamlit/config.toml; \
    else \
        printf "[browser]\ngatherUsageStats = false\n" > /tmp/.streamlit/config.toml; \
    fi; \
    # force la valeur à false si la clé existe déjà
    sed -i 's/^\s*gatherUsageStats\s*=.*/gatherUsageStats = false/g' /tmp/.streamlit/config.toml || true; \
    # ajoute la clé si absente
    grep -q "gatherUsageStats" /tmp/.streamlit/config.toml || \
      printf "\n[browser]\ngatherUsageStats = false\n" >> /tmp/.streamlit/config.toml

EXPOSE 7860

# Lance Streamlit (port injecté par la plateforme, fallback 7860 en local)
CMD streamlit run app.py --server.address=0.0.0.0 --server.port ${PORT:-7860}
