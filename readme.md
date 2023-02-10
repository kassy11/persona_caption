## Persona Caption Dialog

人物画像を使った、なりきりTelegramボット

## 実行

1. BotFatherを用いてTelegramトークンを取得する
2. 環境変数`TOKEN`にTelegramトークンを設定する: `export TOKEN=<YOUR TELEGRAM BOT TOKEN>`
3. JPersonaChatでファインチューニング済みGPT2モデルを`GPT2/model/`以下に配置する
   1. ファインチューニングには[kassy11/convai_jpersona: ConvAI finetuned by JPesonaChat](https://github.com/kassy11/convai_jpersona)を利用してください
4. [chiVe](https://github.com/WorksApplications/chiVe)のgensimデータをダウンロードし、`.kv`ファイルと `.npy`ファイルを`data/chive`以下に配置する
5. Telegramボットを起動: `python bot.py`
6. Telegramボットに`/start`と送信し、ボットを起動する

## アーキテクチャ

人物画像からその人物のペルソナを推測して出力する「ペルソナキャプション生成」モジュールと、出力されたペルソナをもとに雑談対話を行う「ペルソナ対話」モジュールから構成されています。

### ペルソナキャプション生成

<img src="./images/persona_caption.png" title="ペルソナキャプション生成" width="800">

### ペルソナ対話

<img src="./images/persona_dialog.png" title="ペルソナ対話" width="800">

## 対話例

<img src="./images/chat_example1.png" title="対話例1" width="700">
<img src="./images/chat_example2.png" title="対話例2" width="500">
