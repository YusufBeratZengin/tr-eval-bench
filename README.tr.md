# tr-eval-bench

**Türkçe metin gömme (embedding) ve RAG geri getirme modellerini CPU'da, önbellekteki modellerle tek komutla bir dakikadan kısa sürede puanlayın.**

Her Türkçe RAG projesinin ilk sorusu aynı: *hangi gömme modelini kullanmalıyım?* `tr-eval-bench`
bu soruya hızlı cevap vermek için yazıldı. Paketin içinde gelen 100 çiftlik anlamsal benzerlik
seti ile 100 soruluk / 100 belgelik geri getirme seti üzerinde Spearman, NDCG@10, MRR@10 ve
Recall@10 hesaplar. Gigabaytlarca veri indirmek ya da GPU kiralamak gerekmez.

```
$ tr-eval-bench "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
dataset  tr-eval-mini-0.1.0
data     STS 100 pairs | retrieval 100 queries / 100 passages

TASK          METRIC     SCORE
------------  ---------  ------
STS-TR        Spearman   0.8538
STS-TR        Pearson    0.8865
Retrieval-TR  NDCG@10    0.9457
Retrieval-TR  MRR@10     0.9348
Retrieval-TR  Recall@10  0.9800

elapsed  11.68s
```

Bu çıktı gerçek bir koşudan: AMD Ryzen 9 5950X, 5 vCPU, GPU yok, Python 3.14, model Hugging Face
önbelleğinde hazır. Modeli ilk kez indirirken süre uzar; ölçümün kendisi model yüklendikten sonra
yaklaşık 2 saniye sürer.

## Kurulum

```bash
pip install tr-eval-bench          # çekirdek: yalnızca NumPy + PyArrow
pip install "tr-eval-bench[hf]"    # Hugging Face modelleri için sentence-transformers
```

Python 3.10+; Linux, macOS ve Windows'ta çalışır. Yerel modeller için API anahtarı gerekmez.

## Neden var?

- Akademik Türkçe kıyaslama seti [`selmanbaysan/mteb_tr`](https://github.com/selmanbaysan/mteb_tr)
  kapsamlı ama gigabaytlarca veri indiriyor, GPU istiyor ve saatler sürüyor. Hızlı model seçimi ya
  da CI kontrolü için ağır kalıyor.
- Hugging Face'teki [MTEB Türkçe liderlik tablosu](https://huggingface.co/spaces/alibayram/mteb-turkish)
  büyük ve bilinen modelleri izliyor; kendi küçük fine-tune'larını deneyen ekiplerin yeni bir
  checkpoint'in gerçekten iyileşip iyileşmediğini hızlıca göreceği yerel bir araç yok.
- `tr-eval-bench` kapsamdan feragat edip hızı seçiyor: 200 etiketli örnek, paket içinde, çevrimdışı,
  CPU'da, tek komut, makine okunur çıktı.

MTEB-TR'nin yerine geçmez, onu tamamlar. Ayrıntı için [Sınırlar](#sınırlar-ve-dürüstlük).

## Neyi ölçüyor?

| Görev | Örnek | Metrik | Ne söyler? |
| --- | ---: | --- | --- |
| `sts` — STS-TR | 100 çift (0–5) | Spearman, Pearson | Model, cümle benzerliğini insan sıralamasına yakın sıralıyor mu? |
| `retrieval` — Retrieval-TR | 100 soru / 100 belge | NDCG@10, MRR@10, Recall@10 | Sorunun cevabını taşıyan tek belgeyi bulabiliyor mu? |
| `--hybrid` (isteğe bağlı) | aynı set | NDCG@10, MRR@10, Recall@10 | BM25 + yoğun (dense) birleşimi (RRF) tek başına yoğun aramadan iyi mi? |

Metrikler standart tanımlarla ve elle hesaplanmış değerlere karşı test edilir:

- **Spearman** — ortalama sıraların Pearson korelasyonu (eşit değerlerde sıra ortalanır).
- **NDCG@10** — `DCG = Σ rel_i / log2(i+1)`, doğrusal kazanç, ideal sıralama sorgunun
  etiketlerinden üretilir, `NDCG = DCG / IDCG`.
- **MRR@10** — ilk ilgili sonucun sıra numarasının tersi; sorgular üzerinden ortalama.
- **Recall@10** — ilgili belgelerin ilk 10'da bulunma oranı.
- **Hibrit** — BM25 (`k1=1.5`, `b=0.75`) ile yoğun sıralama, RRF ile birleştirilir
  (`skor = Σ 1/(60 + sıra)`), aynı metriklerle ölçülür.

Tüm skorlar L2-normalize edilmiş vektörlerde kosinüs benzerliğiyle hesaplanır.

## Komut satırı

```bash
tr-eval-bench MODEL [seçenekler]                  # tek modeli değerlendir
tr-eval-bench diff TEMEL ADAY [seçenekler]        # iki modeli metrik metrik karşılaştır
tr-eval-bench datasets [--verify]                 # veri seti sayıları + sağlama toplamları
```

| Seçenek | Varsayılan | Anlamı |
| --- | --- | --- |
| `--backend {auto,hf,api,hashing}` | `auto` | `hashing*` için yerleşik sözlüksel taban, diğerlerinde Hugging Face. |
| `--device` | `cpu` | Hugging Face modelleri için cihaz (`cpu`, `cuda`, `mps`). |
| `--api-url` | `$TR_EVAL_API_URL` veya `http://localhost:11434/v1` | OpenAI uyumlu uç nokta. |
| `--batch-size` | `32` | Gömme yığın boyutu; büyük modellerde RAM yetmezse düşürün. |
| `--tasks` | `sts,retrieval` | Görev alt kümesi. |
| `--limit` | kapalı | Yalnızca ilk N örneği ölç (hızlı deneme). |
| `--hybrid` | kapalı | BM25 + yoğun RRF skorlarını da raporla. |
| `--format {table,json,markdown}` | `table` | Rapor biçimi. |
| `--output YOL` | kapalı | Raporu dosyaya yaz. |
| `--export-markdown YOL` | kapalı | Ayrıca Markdown raporu üret. |
| `--fail-under METRIK=DEGER` | kapalı | Metrik eşiğin altındaysa çıkış kodu 1 (tekrarlanabilir). |
| `--max-drop METRIK=DEGER` | kapalı | `diff` için: aday bu kadar gerilerse çıkış kodu 1. |
| `-q`, `--quiet` | kapalı | Raporu ekrana basma (`--output` ile kullanışlı). |

`--fail-under` / `--max-drop` için kısa adlar: `spearman`, `pearson`, `ndcg`/`ndcg10`,
`mrr`/`mrr10`, `recall`/`recall10`, `hybrid_ndcg`, `hybrid_mrr`.

Çıkış kodları: `0` başarılı, `1` eşik ihlali, `2` yapılandırma/çalışma hatası.

### Yerel API sunucuları (Ollama, vLLM, TGI, llama.cpp)

API koşucusu OpenAI uyumlu `POST /v1/embeddings` protokolünü konuşur. Ollama bunu `/v1` altında
sunuyor; vLLM ve TGI de destekliyor.

```bash
ollama pull nomic-embed-text
tr-eval-bench nomic-embed-text --backend api --api-url http://localhost:11434/v1
```

Sunucu anahtar istiyorsa `TR_EVAL_API_KEY` ortam değişkenini kullanın; anahtar komut satırına
yazılmaz, böylece kabuk geçmişine ve CI günlüklerine düşmez.

### CI'da regresyon kapısı

```bash
# Aday model geri getirmede gerilerse derlemeyi durdur
tr-eval-bench diff eski-model yeni-model --max-drop mrr=0.02 --max-drop ndcg=0.02 --quiet

# Fine-tune mutlak eşiğin altına düştüyse durdur
tr-eval-bench ./checkpoints/modelim --fail-under spearman=0.80 --format json --output skorlar.json
```

## Kütüphane olarak

```python
from tr_eval_bench import evaluate_model, compare_models

sonuclar = evaluate_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(sonuclar.sts_spearman)      # 0.8538...
print(sonuclar.retrieval_mrr10)   # 0.9348...
print(sonuclar.to_dict())         # JSON'a hazır çıktı (schema_version 1.1)

fark = compare_models("hashing:256", "hashing:1024", tasks="retrieval")
print(fark.rows[3].metric, fark.rows[3].delta)
```

`evaluate_model` şu parametreleri alır: `tasks`, `backend`, `device`, `batch_size`, `limit`,
`hybrid`, `rrf_k`, `api_url` ve `runner` (`name`, `dimension` ve `embed(texts, batch_size=...)`
sağlayan herhangi bir nesne).

## Paket içindeki veri setleri

| Dosya | Satır | İçerik |
| --- | ---: | --- |
| `sts_mini.parquet` | 100 | 10 alanda dereceli cümle çiftleri (0–5) |
| `rag_mini.parquet` | 100 | Soru + dereceli qrels (`passage_id`, `relevance`) |
| `rag_corpus.parquet` | 100 | Belge kimliği, başlık, metin, alan |
| `MANIFEST.json` | — | Sürüm, sayılar, SHA-256 sağlama toplamları |

- **Lisans:** MIT; metinler bu proje için genel kültür bilgisinden özgün olarak yazıldı, hiçbir
  dış korpus kopyalanmadı, kişisel veri ve gerçek kişi adı yok.
- **Çevrimdışı:** Parquet dosyaları tekerleğin (wheel) içinde. `tr-eval-bench datasets --verify`
  sağlama toplamlarını denetler; değerlendirme sırasında hiçbir şey indirilmez.
- **Yeniden üretim:** `python scripts/build_datasets.py` (kimlik, puan aralığı, dağılım denetimi
  yapar ve manifesti yeniler).
- Ayrıntılı kaynak notu: [`tr_eval_bench/datasets/PROVENANCE.md`](tr_eval_bench/datasets/PROVENANCE.md).

## Ölçülen sonuçlar (referans makine)

AMD Ryzen 9 5950X, 5 vCPU, GPU yok, Python 3.14, `tr-eval-bench 0.1.0`:

| Model | STS Spearman | NDCG@10 | MRR@10 | Süre |
| --- | ---: | ---: | ---: | ---: |
| `hashing:256` (sözlüksel taban) | 0.7753 | 0.9748 | 0.9674 | 0.2 sn |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.8538 | 0.9457 | 0.9348 | 18.9 sn |
| `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 0.8920 | 0.9795 | 0.9725 | 29.2 sn |

Süre, değerlendirme çağrısının başından itibaren uçtan uca ölçüldü (model yüklemesi dahil);
iki model de referans makinede Hugging Face önbelleğinde hazırdı. Kompakt modelin yükleme sonrası
`elapsed_seconds` değeri 11,7 sn, tablodaki 18,9 sn ise yükleme dahil toplamdır. İlk indirme veya
yüklü bir 5 vCPU konteyner belirgin şekilde daha uzun sürebilir; "bir dakikadan kısa" hedefi
önbellekteki modeller için geçerlidir, ilk indirmeler için değil.

> Sözlüksel tabana dikkat: geri getirme soruları olgusal ve altın belgeyle kelime paylaşıyor; bu
> yüzden `hashing` orada rekabetçi kalırken STS'de belirgin şekilde geride kalıyor. İyi modelleri
> ayırmak için `sts`, bozuk boru hatlarını yakalamak için `retrieval` kullanın — bkz.
> [Sınırlar](#sınırlar-ve-dürüstlük).

## Bağımlılıklar ve gerekçeleri

| Paket | Rol | Neden standart kütüphane değil? |
| --- | --- | --- |
| `numpy` | benzerlik matrisleri, sıra istatistikleri | 100×768 matrislerde saf Python 20–50 kat yavaş ve hataya açık. |
| `pyarrow` | paket içi Parquet dosyalarını okumak | Parquet kompakt, tipli ve sağlama toplamlı; JSON kopyası veriyi ikiye çıkarırdı. |
| `sentence-transformers` (ek) | yerel Hugging Face modelleri | Yalnızca `hf` arka ucu için gerekir; `api` ve `hashing` onsuz çalışır. |

Geri kalan her şey (CLI, HTTP, JSON) standart kütüphaneden.

## Sınırlar ve dürüstlük

- **Mini, resmî değil.** 100 STS çifti ve 100 geri getirme sorusu hızlı bir sinyaldir; MTEB veya
  TR-MTEB sıralamasını yeniden üretmez ve öyle alıntılanmamalıdır.
- **Geri getirme doygun.** Korpus küçük ve sorular olgusal olduğu için çalışan her model yüksek
  skor alıyor (MRR@10 ≈ 0.93–0.98) ve sözlüksel taban rekabetçi kalıyor. Bu görevi bozuk boru
  hatlarını ve regresyonları yakalamak için kullanın; iyi modelleri ayırmak için `sts`, `--hybrid`
  ve `diff` daha bilgilendirici. Zor negatifli korpus yol haritasının ilk maddesi.
- **Yalnızca Türkçe.** Paket içi veri Türkçe; kodun kendisi dilden bağımsız.
- **Tek turlu soru-cevap.** Çok adımlı, uzun belge ve diller arası geri getirme henüz yok.
- **API protokolü.** Yalnızca OpenAI uyumlu `/v1/embeddings`; Ollama'nın yerel `/api/embeddings`
  uç noktası desteklenmiyor.
- **Süreler** CPU'ya, model boyutuna ve modelin önbellekte olup olmamasına göre değişir.

## Geliştirme

```bash
git clone https://github.com/YusufBeratZengin/tr-eval-bench
cd tr-eval-bench
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
bash scripts/check.sh        # ruff lint + biçim, mypy, %85 kapsam eşikli pytest
```

Testler ağa çıkmaz: API koşucusu yerel bir HTTP sunucusuna karşı, Hugging Face modelleri ise sahte
modülle sınanır. Ayrıntılar [CONTRIBUTING.md](CONTRIBUTING.md).

## Yol haritası

- 0.2 — zor negatifli geri getirme korpusu, sınıflandırma ve kümeleme mini görevleri
- 0.3 — reranker (cross-encoder) değerlendirmesi, TR→EN diller arası geri getirme
- Sonrası — yanıltıcı bağlam seti, gömme kuantizasyon/gecikme profili, HF Space JSON çıktısı

## Lisans

MIT © 2026 Yusuf Berat Zengin. Ayrıntı: [LICENSE](LICENSE).
