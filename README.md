# dummyMozart

MIDI dosyalarından uzman davranışını öğrenip yeni müzik üreten bir AIRL + PPO projesi.

El ile ödül tasarlamak yerine, `midis/` altındaki parçalardan geçişler çıkarılır. Ayrımcı (AIRL) uzman ile ajanı ayırt eder; PPO bu ödülle bir sonraki token’ı seçmeyi öğrenir. Eğitilen politika yeni bir MIDI dosyası üretir.

## Nasıl çalışır

1. MIDI dosyaları REMI tokenizer ile token dizisine çevrilir.
2. Her pencere `(durum, aksiyon, sonraki durum)` uzman geçişi olur: durum son 64 token, aksiyon bir sonraki token.
3. Ortam (`MIDIMusicEnv`) bir kaydırma penceresidir; aksiyon pencerenin sonuna eklenir.
4. Her AIRL turunda ajan yörüngeleri toplanır, ayrımcı eğitilir, PPO `f_θ` ödülüyle güncellenir.
5. `compose.py` checkpoint’ten token üretir ve MIDI’ye çevirir.

```
midis/*.mid  →  uzman geçişleri  →  AIRL (ayrımcı + PPO)  →  checkpoints/ppo.pt  →  output/*.mid
```

## AIRL sözde kod

Eğitim `airl.py` içindeki bu döngüyü izler. Ayrımcı uzmanı ajan’dan ayırt etmeyi öğrenir; PPO ise ayrımcının ürettiği ödülle politikayı günceller.

```
Uzman geçişlerini yükle: (s, a, s') ~ τ_E
Ayrımcı D_θ = {g, h} ve politika π'yi başlat

her AIRL iterasyonunda:
    Ajan geçişlerini topla: (s, a, s') ~ π

    her ayrımcı epoch'unda:
        Uzman ve ajan batch'i örnekle
        Her iki batch için log π(a|s) hesapla
        f_θ(s, a, s') = g(s, a) + γ · h(s') − h(s)
        logits = f_θ − log π
        Loss = BCE(uzman, 1) + BCE(ajan, 0)
        Ayrımcıyı güncelle

    Ödül: r(s, a, s') = f_θ(s, a, s')
    Politikayı PPO ile bu ödülü kullanarak güncelle
```

- `g(s, a)` öğrenilen durum-aksiyon ödülü, `h(s)` potansiyel (shaping) fonksiyonudur.
- `f_θ` shaped ödüldür; `h` terimleri politika optimumunu değiştirmeden öğrenmeyi kolaylaştırır.
- Logits’ten `log π` çıkarılması, ayrımcının “bu geçiş uzman mı?” sorusunu politikadan bağımsız sormasını sağlar.
- Uzman 1, ajan 0 etiketlenir (BCE). PPO ortam ödülü olarak `f_θ` kullanır.

## Kurulum

Python 3.10+ önerilir.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Uzman MIDI dosyalarını `midis/` klasörüne koyun (alt klasörler taranır). En az `SEQ_LEN + 1` token üretecek kadar uzun dosyalar gerekir.

## Eğitim

```bash
python train.py
python train.py --iters 50
```

Varsayılan iterasyon sayısı `config.N_AIRL_ITERS` (100). Her turda:

- ajan `AGENT_COLLECT_STEPS` adım toplar
- ayrımcı `DISC_EPOCHS` epoch eğitilir
- PPO `PPO_TIMESTEPS` adım güncellenir
- checkpoint `checkpoints/ppo.pt` olarak kaydedilir

## Beste üretme

```bash
python compose.py
python compose.py --steps 256
python compose.py --seed-midi midis/ornek.mid --output output/parca.mid
```

| Argüman | Açıklama |
|---|---|
| `--checkpoint` | PPO ağırlıkları (varsayılan: `checkpoints/ppo.pt`) |
| `--steps` | Üretilecek token sayısı (varsayılan: 128) |
| `--seed-midi` | Bağlam için kullanılacak MIDI (en az 64 token) |
| `--output` | Çıktı yolu (yoksa `output/compose_<zaman>.mid`) |

## Dosyalar

| Dosya | Görev |
|---|---|
| `train.py` | AIRL döngüsü |
| `compose.py` | Checkpoint’ten MIDI üretimi |
| `config.py` | Yollar, tokenizer, AIRL ve PPO hiperparametreleri |
| `dataset.py` | MIDI → uzman geçişleri |
| `midi_io.py` | REMI encode / decode |
| `env.py` | Gymnasium kaydırma-pencere ortamı |
| `airl.py` | Ayrımcı, ödül sarmalayıcı, yörünge toplama |
| `ppo.py` | Actor-critic, GAE, clipped PPO |

## Ayarlar

Önemli değerler `config.py` içindedir:

- `MIDI_DIR`, `SEQ_LEN` (64), `COUNTRY_FILTER` — veri
- `N_AIRL_ITERS`, `DISC_LR`, `BATCH_SIZE` — AIRL
- `PPO_LR`, `PPO_CLIP`, `PPO_ENT_COEF` — politika

Ülkeye göre filtre: `COUNTRY_FILTER = "England"` yalnızca `midis/England/` altını kullanır.

## Gereksinimler

- gymnasium
- torch
- numpy
- miditok
- tqdm
- stable-baselines3 (kurulumda listelenir; eğitim kendi PPO uygulamasını kullanır)
