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
- ~~CSD (Children's Song Dataset, KAIST)~~ / ~~GTSinger~~: 当初これらを代替候補として挙げたが、**どちらもCC BY-NC-SA 4.0(非商用限定)であることが判明し、撤回**した(issue #2のコメント参照)。ユーザーの前回記事のPixabayトラックと同様、**商用利用可(Pixabayのcontent licenseは商用利用可・クレジット表記不要)**が実歌唱コーパスの必須条件のため、この2つは使えない。

### 追加調査: Pixabay Musicから追加トラックを探す試み(ブロックされた)

商用利用可という制約を満たす実歌唱ソースとして、既存のPixabayトラックに加えて追加のPixabay Music楽曲(ボーカル入り・歌詞入りのもの)を探そうとしたが、**pixabay.com自体がCloudflareのbot対策(JSチャレンジ)によってサイト全体で自動アクセスをブロックしている**ことを確認した。

- `curl`(ブラウザ風User-Agent付き)、`WebFetch`のどちらでも `pixabay.com/` 直下から `robots.txt` に至るまで一貫して403(`cf-mitigated: challenge`)。
- Pixabayの公式API(`pixabay.com/api/docs/`)は画像・動画のみが対象で、音楽(Music)は含まれていない。
- Web検索(Claude組み込みのWebSearch、別経路でインデックスされたスニペットを取得)経由では、ボーカル入りの候補トラックのタイトル・ページURLをいくつか特定できた(例: [`KI - Song (Pop, Vocals)`](https://pixabay.com/music/pop-ki-song-pop-vocals-378912/)、"Dancing Isn't Asking"、"Apocalypse (1) - Original Lyrics" など)。ただし実際のダウンロード用CDN URL(`cdn.pixabay.com/download/audio/...`)はページ本体からしか取得できず、そのページ自体が上記の理由でエージェントからは開けない。
- web.archive.org経由での代替取得も試したが、こちらもツールから利用不可/レート制限で失敗。

**結論**: **自動化ツールによるPixabayの新規トラック発掘は、このセッション環境では実行不可能(ハードブロッカー)。** これはDAMP-VPBとは異なる種類のブロッカー(承認待ちではなく、bot対策そのもの)だが、結果として同様に「ユーザー本人の手作業が必要」という結論になる。上記のトラック候補URLは人間が普通にブラウザで開けば(bot判定に引っかからないはず)問題なく開けるはずなので、ユーザー側で目視確認の上ダウンロードしてもらうのが現実的。ボット対策を能動的に回避する手段(UA偽装以上のもの、プロキシ経由など)は意図的に試みていない。

### フォールバック: 既存の1曲内でセグメント数を増やす

新曲の発掘がブロックされたため、方針を「複数曲」から「既存の1曲(Pixabay, 商用利用可, 前回記事でも使用)内でセグメント数を増やす」に切り替えた。前回は代表5区間のみだったが、今回はwhisperの書き起こし全27区間すべてでbaseline / wav2vec2-jaを実行し、kanasimでリファレンスと比較した(`scripts/build_song_corpus.py`)。

| | baseline | wav2vec2-ja |
|---|---|---|
| 平均kanasim距離 (有効値のみ) | 88.15 (n=25) | 105.13 (n=24) |
| 区間ごとの勝敗(距離が小さい方) | **15/27** | 12/27 |

**前回(5区間)の「baselineが一貫して勝つ」という結論は、27区間に増やすと弱まった。** 平均では引き続きbaselineが上回るものの、個々の区間で見るとwav2vec2-jaが勝つケースも12/27(44%)あり、5区間だけでは見えなかった「どちらが良いかは区間による」という実態が見えてきた。

もう一つの発見として、**曲の後半(2番以降)ほど両手法とも認識が大きく崩れる傾向**があった。特に18〜25番目の区間(歌の中盤〜終盤、"Oh eternal light"のサビが2回目に登場する箇所を含む)では、baseline/wav2vec2-jaともに距離100超、時にはbaselineの出力が空になる(25番目)ケースも発生した。同じ歌詞のサビが1回目(10番目, "ス"の1文字)と2回目(22番目, "ツ"の1文字)でどちらも壊滅的に崩れていたのは、前回のボーカル分離検証の結果と合わせて考えると、この特定のフレーズの歌い方(伸ばし方・ピッチ)自体がこのモデル構成にとって鬼門である可能性を裏付けている。

## 4. ボーカル分離(Demucs)の有効性調査

前回の検証で"Oh eternal light, that shines so pure"の区間がbaselineで`ス`の1文字にまで崩壊していた件について、**伴奏(BGM)の干渉が原因か、歌唱表現自体が原因か**を切り分けるため、Demucs (htdemucs, 2-stems) でボーカルを分離してから同じ区間を再認識した(`scripts/vocal_separation_check.py`)。

- **環境構築で複数の依存関係の壁にぶつかった**(この環境固有の事情として記録): (1) torch 2.9.0とtorchaudio 2.11.0のABI不一致(`undefined symbol`)→ torchaudioをtorchと同系列の2.9.0に固定して解消。(2) 新しいtorchaudioはCLIの音声読み込みに`torchcodec`(ffmpeg共有ライブラリ必須)を要求し、この環境にはffmpegがなく失敗 → demucsのCLIサブプロセスを使うのをやめ、`demucs.apply.apply_model` / `demucs.pretrained.get_model` のPython APIを直接呼び出し、音声読み込み自体はlibrosa/soundfile(libsndfileが直接mp3を読めるためffmpeg不要)で行うことで回避した。CPUで3分强の楽曲の分離は約47秒。
- **分離自体は問題なく動いた**(実行時間・リソース面で本番投入も現実的)。
- **ただし、分離後も同じ区間の認識は改善しなかった**。分離前: `ス`(1文字)。分離後(ボーカルのみ): `ンンツ`。IPA出力を見ると `x ou5 i5 s. n n ai5 t ai5 s. ɑ5 s ou5 ph u5` のように、末尾に数字が付いた記号(`ou5`, `ai5`, `ɑ5`など)が混ざっている。これはwav2vec2-espeakモデルが英語ではなく声調言語(トーン付き)の音素として認識してしまっている兆候で、分離後の音声(位相ずれ・アーティファクトを含みがちなDemucsの出力の癖)がモデルを混乱させている可能性がある。

**結論**: この区間の崩壊は、少なくとも単純なBGM除去だけでは解決しない。原因は伴奏の干渉というより、**サステインする音・ピッチが大きく動く歌唱表現そのもの、もしくは分離後の音声アーティファクト**によるものである可能性が高い。ボーカル分離は前処理として有効な場合もありうるが、今回の代表的な失敗例に関しては効果が確認できなかった、と正直に記録しておく。

## 次にやるべき具体的なこと(#2の次の一歩の提案)

1. **商用利用可な実歌唱トラックをユーザー本人に何件か集めてもらう。** Pixabayは自動化ツールからはbot対策でブロックされているため、これはユーザーの手作業が必要。上記で見つけた候補ページ(`KI - Song (Pop, Vocals)`など)を起点に、通常のブラウザで"vocal pop"・"english lyrics"などのタグで探してもらうのが早い。他に商用利用可なボーカル素材サイト(Pixabayと同様の無料音源サイトなど)も候補に入れてよい。
2. トラックが集まるまでの間は、**既存の1曲・27区間のデータ(`local/song_corpus/results.json`)を「実歌唱の暫定検証セット」として使い、疑似ラベル(TTS 120件)と組み合わせて学習データ構成の設計を先に進める**(#3 モデル学習の前提として)。1曲だけでもbaseline/wav2vec2-jaの傾向(平均では僅差、区間によっては逆転する)は掴めている。
3. 音声拡張は「一様なピッチシフト/テンポ変更」だけでなく、母音部分だけを狙って伸ばす・ピッチ変調するような、より歌唱に忠実な拡張を検討する(GTSinger自体は非商用ライセンスのため学習データには使えないが、技法ラベルの設計思想は参考にできる)。
4. ボーカル分離は「前処理として常に使うもの」ではなく、**選択肢の一つとして評価に含める**(分離あり/なしの両方でモデルを評価し、実際に効果があるか確認する)程度の位置づけに留める。
5. 曲の後半で認識が崩れやすい傾向が見えたので、**学習データも曲の冒頭に偏らないよう、サビ・Bメロなど曲構造の異なる箇所を意識してバランスよく集める**とよさそう。

## 参考

- [docs/kana-asr-experiment/RESULTS.md](https://github.com/jiroshimaya/songasr/blob/docs/kana-asr-experiment/docs/kana-asr-experiment/RESULTS.md)(branch: `docs/kana-asr-experiment`。このブランチには存在しないので注意)
- [DAMP-VPB (Zenodo)](https://zenodo.org/records/2616690) — 商用利用不可・要申請
- ~~CSD~~ / ~~GTSinger~~ — CC BY-NC-SA 4.0のため不採用(issue #2コメントで訂正済み)
- [Pixabay content license summary](https://pixabay.com/service/license-summary/) — 商用利用可・クレジット不要
- [27区間ぶんの結果 (song_corpus_results.json)](./song_corpus_results.json)
- [Demucs](https://github.com/facebookresearch/demucs)
