# 正解データ作成の検討ログ (issue #2)

[トラッキングissue #5](https://github.com/jiroshimaya/songasr/issues/5) の3ステップのうち、[#2 正解データ作成](https://github.com/jiroshimaya/songasr/issues/2)に着手した記録。docs/kana-asr-experiment/RESULTS.md (branch: `docs/kana-asr-experiment`) の続き。専用モデルを学習する前段階として、(1) 疑似ラベル方式のデータ生成、(2) 歌唱を模した音声拡張、(3) 実歌唱コーパスの入手可否調査、(4) ボーカル分離の有効性調査、の4点を検証した。モデル学習(#3)には着手していない。

## 1. 疑似ラベル方式でのデータセット生成

`docs/kana-asr-experiment/RESULTS.md` で使ったbaselineパイプライン(wav2vec2-espeak IPA→ARPABET→arpakana、`src/songasr/kana_baseline.py` として本体に移植)を疑似ラベラーとして使い、(音声, カナ)ペアを規模を上げて生成した。

- **文のソース**: 実在の楽曲の歌詞をそのまま使うと著作権上の懸念があるため、`scripts/kana_dataset_sentences.py` でポップス歌詞によくあるパターン(主語+動詞句+副詞句、縮約形、子音クラスタを含む文など)をテンプレートで組み合わせ、オリジナルの英語短文を機械生成した。継続比較のため、TTS実験(RESULTS.md)で使った5文もそのまま含めている。著作権的にクリーンな一方、実際の歌詞の言い回しの多様性はカバーしきれていない。
- **規模**: 120文(`scripts/build_kana_dataset.py --n 120`)。各文をgTTSで読み上げ音声化し、baselineのIPA→ARPABET→カナを疑似ラベルとして付与。全120件でカナラベルが空になったケースはなし。
- **重要な注意点(弱教師あり)**: これは**弱教師あり(weak supervision)**であり、ラベルの質はbaseline自体の精度に依存する。`docs/kana-asr-experiment/RESULTS.md` で示した通り、baselineは(標準的な外来語カナ表記との比較で)kanasim距離平均50前後の誤りを含む。つまりこのデータセットで学習したモデルは、baselineの癖・誤りごと学習してしまうリスクがある。あくまで「音声とそれらしいカナのペアを大量に用意する」という規模の確保が目的で、正解データの質の担保にはなっていない。
- 出力は `local/kana_dataset/`(gitignore対象、120件・音声15MB程度)に生成。サンプルとして先頭10件を [`manifest_sample.jsonl`](./manifest_sample.jsonl) にコミットしている。

## 2. 歌唱っぽい音響への水増し(augmentation)の試作

TTS音声と実歌唱の差(前回検証で確認したwav2vec2-jaの精度劣化: 同じ歌詞で79.3→96.3)を安く埋められないか、librosaの`pitch_shift`/`time_stretch`を試した(`scripts/augment_singing.py`)。s1_moon ("Under the moon and the quiet sky so clear") に対する結果:

| variant | duration | mean F0 | baselineのカナ出力 |
|---|---|---|---|
| original | 3.67s | 279 Hz | アンダーダムーナンダクワイイッツカイソウクリ |
| pitch +4半音 | 3.67s | 349 Hz | ントゥムールナアイスサー |
| pitch −4半音 | 3.67s | 221 Hz | アナムーニンダクワイサイソウクリ |
| time-stretch 0.7x(遅く) | 5.25s | 276 Hz | アンダーダムーナンドダクワイイッツカイゾウクリ |
| time-stretch 1.3x(速く) | 2.83s | 277 Hz | インダムーニンダクワイツァイソウフィ |

(聴感でのチェックはできないため、`librosa.pyin`で推定した平均F0とスペクトル重心を代わりに確認し、意図通りの変化になっていることは確認した。)

**わかったこと**: ピッチシフトの方がtime-stretchよりbaselineの認識を大きく崩す(特に+4半音で結果が大きく変わった)。逆にtime-stretchだけなら(0.7倍速でも)ほぼ元の認識結果を保っていた。

**この方法の限界**: 実際の歌唱で起きているのはピッチシフトやテンポ変更のような音源全体への一様な変換ではなく、

- 特定の**母音だけを長く伸ばす**(子音はそのまま、単語全体が均等に伸びるわけではない)
- 伸ばした音の中で**ビブラート**(周期的な微小ピッチ変動)がかかる
- **メリスマ**(1音節が複数の音高にまたがる)

といった、より局所的・非一様な変化。今回試した一様なピッチシフト/テンポ変更は「歌声っぽい何か」にはなるが、実歌唱で問題になっている現象(前回検証で"ス"の1文字に崩壊したケースなど)を再現できているとは言い切れない。安価な水増し手段としては足がかりになるが、**この水増しだけでは実歌唱のギャップを十分に埋められない可能性が高い**、というのが現時点の結論。

## 3. 実歌唱コーパスの入手可否調査

前回の検証(#4 検証相当)でTTS音声と実歌唱の間にギャップがあることが分かったため、教師データを実歌唱ベースにできないか調査した。

- **DAMP-VPB (Smule Vocal Performances Balanced)**: Zenodoで公開されている(https://zenodo.org/records/2616690, 24,874件のソロ歌唱)が、**「restricted access」でアクセス申請フォームの提出とSmuleのResearch Data License Agreementへの同意が必要**。申請後の承認プロセスは人間の審査を挟むため、エージェントのセッション内で完結できない**ハードブロッカー**。ユーザー自身が個人アカウントで申請・承認を待つ必要がある。
- **CSD (Children's Song Dataset, KAIST)**: 英語50曲・韓国語50曲、各2つのキーで歌われたプロ歌手の歌唱データ。MIDI・歌詞(グラフェム/音素レベル)付き。CC BY-NC-SA 4.0、**Zenodo/GitHub経由で申請なしに直接ダウンロード可能**。英語の歌詞付き歌唱データとして今回の用途に合致する。
- **GTSinger**: 9言語(英語含む)対応の大規模歌唱コーパス。Hugging Face (`GTSinger/GTSinger`など複数ミラー) で**無料・申請なしで直接ダウンロード可能**。TextGridで単語/音素境界、ビブラート・メリスマなどの技法ラベルも付与されており、まさに今回課題になっている現象(メリスマ・ビブラート)がラベル済みなのが強み。

**結論**: DAMP-VPBは今回のようなエージェントセッションでは入手できない(ユーザーによる個別申請が必要)。一方、**CSDとGTSingerは英語歌唱+歌詞付きで即座にダウンロード可能**であり、実歌唱データの入手可否という論点については「取れないわけではない、CSD/GTSingerが現実的な代替」というのが結論。次のステップではこのどちらかを実際に使う想定で進めるのが良さそう。

## 4. ボーカル分離(Demucs)の有効性調査

前回の検証で"Oh eternal light, that shines so pure"の区間がbaselineで`ス`の1文字にまで崩壊していた件について、**伴奏(BGM)の干渉が原因か、歌唱表現自体が原因か**を切り分けるため、Demucs (htdemucs, 2-stems) でボーカルを分離してから同じ区間を再認識した(`scripts/vocal_separation_check.py`)。

- **環境構築で複数の依存関係の壁にぶつかった**(この環境固有の事情として記録): (1) torch 2.9.0とtorchaudio 2.11.0のABI不一致(`undefined symbol`)→ torchaudioをtorchと同系列の2.9.0に固定して解消。(2) 新しいtorchaudioはCLIの音声読み込みに`torchcodec`(ffmpeg共有ライブラリ必須)を要求し、この環境にはffmpegがなく失敗 → demucsのCLIサブプロセスを使うのをやめ、`demucs.apply.apply_model` / `demucs.pretrained.get_model` のPython APIを直接呼び出し、音声読み込み自体はlibrosa/soundfile(libsndfileが直接mp3を読めるためffmpeg不要)で行うことで回避した。CPUで3分强の楽曲の分離は約47秒。
- **分離自体は問題なく動いた**(実行時間・リソース面で本番投入も現実的)。
- **ただし、分離後も同じ区間の認識は改善しなかった**。分離前: `ス`(1文字)。分離後(ボーカルのみ): `ンンツ`。IPA出力を見ると `x ou5 i5 s. n n ai5 t ai5 s. ɑ5 s ou5 ph u5` のように、末尾に数字が付いた記号(`ou5`, `ai5`, `ɑ5`など)が混ざっている。これはwav2vec2-espeakモデルが英語ではなく声調言語(トーン付き)の音素として認識してしまっている兆候で、分離後の音声(位相ずれ・アーティファクトを含みがちなDemucsの出力の癖)がモデルを混乱させている可能性がある。

**結論**: この区間の崩壊は、少なくとも単純なBGM除去だけでは解決しない。原因は伴奏の干渉というより、**サステインする音・ピッチが大きく動く歌唱表現そのもの、もしくは分離後の音声アーティファクト**によるものである可能性が高い。ボーカル分離は前処理として有効な場合もありうるが、今回の代表的な失敗例に関しては効果が確認できなかった、と正直に記録しておく。

## 次にやるべき具体的なこと(#2の次の一歩の提案)

1. **CSD (Children's Song Dataset) の英語パートを実際に取得し、疑似ラベル方式(baseline出力)とdocs/kana-asr-experimentで使った手書きリファレンスの両方でざっくり評価してみる。** DAMP-VPBはユーザー本人による申請待ちになるため、まずCSDで手を動かすのが早い。
2. 疑似ラベル(TTS 120件)とCSD由来の実歌唱データを混ぜて学習データを構成する方針を固める(#3 モデル学習の前提として)。
3. 音声拡張は「一様なピッチシフト/テンポ変更」だけでなく、GTSingerのTextGridラベル(ビブラート・メリスマの区間情報)を参考に、母音部分だけを狙って伸ばす・ピッチ変調するような、より歌唱に忠実な拡張を検討する。
4. ボーカル分離は「前処理として常に使うもの」ではなく、**選択肢の一つとして評価に含める**(分離あり/なしの両方でモデルを評価し、実際に効果があるか確認する)程度の位置づけに留める。

## 参考

- [docs/kana-asr-experiment/RESULTS.md](https://github.com/jiroshimaya/songasr/blob/docs/kana-asr-experiment/docs/kana-asr-experiment/RESULTS.md)(branch: `docs/kana-asr-experiment`。このブランチには存在しないので注意)
- [DAMP-VPB (Zenodo)](https://zenodo.org/records/2616690)
- [CSD: Children's Song Dataset (KAIST MAC Lab)](https://mac.kaist.ac.kr/resources.html)
- [GTSinger (Hugging Face)](https://huggingface.co/datasets/GTSinger/GTSinger)
- [Demucs](https://github.com/facebookresearch/demucs)
