# Pythonの公式イメージをベースイメージとして使用
FROM python:3.10-slim

# 環境変数を設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をインストール
# requirements.txtを先にコピーしてキャッシュを活用
COPY requirements.txt .
RUN pip install -r requirements.txt

# sentence-transformersのモデルを事前にダウンロードするためにキャッシュディレクトリを作成・権限付与
# (HuggingFaceのデフォルトキャッシュディレクトリ)
RUN mkdir -p /root/.cache/huggingface/sentence_transformers && \
    chmod -R 777 /root/.cache/huggingface

# プロジェクトのソースコードをコピー
COPY . .

# コンテナ起動時に実行するコマンド
CMD ["python", "main.py"]
# main.py がまだないので、一旦コンテナが起動し続けるようにする
# CMD ["tail", "-f", "/dev/null"]