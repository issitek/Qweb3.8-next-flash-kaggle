# Qwen3.8-Flash-Next Coder IQ1_M en Kaggle 2x T4 — optimizado

Versión optimizada de [jigs10/qwen3.8-flash-strata-kaggle](https://github.com/jigs10/qwen3.8-flash-strata-kaggle).

Mismo modelo, mismo hardware (Kaggle gratis, **GPU T4 x2**) y el mismo motor
([Strata](https://github.com/Niko1221/Strata)), pero con **MTP encendido** y el resto de ajustes puestos a favor
de la velocidad. El notebook mide el resultado y lo imprime; no hay cifras inventadas.

---

## Qué cambia respecto al original

| | Original | Este repo |
|---|---|---|
| **MTP** | `--mtp off` | **`--mtp on`** + `--spec 4 --spec-min-p 0.5` + `--mtp-q4 all` |
| **Expertos** | se leen del SSD cuando no caben | `--resident-budget-gib` los copia a RAM: cero lecturas de disco al responder |
| **Arranque** | `Popen` + `sleep(60)` + leer el log a mano | `strata launch --no-tui --yes`: descarga, sirve y devuelve el control |
| **Re-ejecución** | lanzar dos veces pelea por el puerto 8080 | idempotente: si ya hay servidor, lo reutiliza |
| **Disco** | mirar el log y matar antes de 85 GB | guarda automática: aborta antes de llenar el disco |
| **RAM** | sin control | guarda automática: aborta antes del OOM |
| **Validación** | leer el plan a ojo | valida el plan **y** los flags reales del run config |
| **Benchmark** | una pasada | calentamiento + 3 pasadas, **mediana** |
| **Tuning** | `--tune skip` | igual (en Kaggle no hay tiempo para 20 min de tuning), pero documentado |
| **Acceso** | solo `127.0.0.1` | túnel de **Cloudflare** con API key (la URL es pública) |

## Por qué MTP es la palanca grande aquí

El pack IQ1_M pesa ~58 GB. Kaggle da **32 GB de VRAM** (2 × T4 de 16 GB) y **~31 GB de RAM**, así que el modelo
no cabe entero en la GPU: el motor funciona en modo híbrido y lee pesos desde RAM en cada paso. Eso significa que
el decode está limitado por **ancho de banda de memoria**, no por cómputo.

Con MTP (multi-token prediction) el motor propone hasta 4 tokens y el modelo completo los verifica en la misma
pasada de pesos. Resultado: más tokens por cada lectura de pesos, que es exactamente el cuello de botella aquí.

Dos detalles que el notebook aprovecha y que el original desaprovechaba:

- **`--mtp` ya viene encendido por defecto en Strata.** Apagarlo ahorra ~1 GB de VRAM pero es más lento
  ([`docs/TERMINAL_LAUNCH.md`](https://github.com/Niko1221/Strata/blob/main/docs/TERMINAL_LAUNCH.md)).
- **MTP no cambia las respuestas.** Solo propone tokens; el modelo completo verifica cada uno. Es velocidad gratis
  en calidad ([`docs/MAC.md`](https://github.com/Niko1221/Strata/blob/main/docs/MAC.md)).

La justificación flag por flag, con la cita exacta de la documentación de Strata, está en
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md).

## Cómo usarlo

1. Kaggle → **File > Import Notebook** → sube `strata_kaggle.ipynb`.
2. A la derecha: **Accelerator: GPU T4 x2** e **Internet: On**.
3. Ejecuta las celdas en orden (Run All).

| Sección | Qué hace |
|---|---|
| 1 | Configuración: todos los ajustes en un solo sitio |
| 2 | Preflight: GPUs, RAM, disco, CPUs |
| 3 | Instala Strata solo si falta |
| 4 | Instala `cloudflared` para el túnel (idempotente) |
| 5 | `--plan` y validación (**debe decir Coder IQ1_M**) |
| 6 | Arranque con guardas de disco y RAM |
| 7 | Verifica que MTP y los ajustes están aplicados de verdad |
| 8 | URL pública del túnel y API key (enmascarada) |
| 9 | Ejemplo de cliente (OpenAI SDK / curl) |
| 10 | Benchmark: TTFT y tok/s (mediana de 3) |
| 11 | Opcional: SVG con vista previa en vivo |
| 12 | Para el servidor y el túnel |

La primera ejecución descarga ~58 GB del modelo y ~6 GB del draft MTP, así que tarda. Las siguientes sesiones de
Kaggle empiezan de cero y vuelven a descargar (ver *Límites de Kaggle*).

## Ajustes

Todo está en el diccionario `CONFIG` de la sección 1. También se puede cambiar sin editar código con variables `SK_*`:

| Ajuste | Valor | Por qué |
|---|---|---|
| `family` / `model` | `coder` / `iq1_m` | Es el único pack que cabe en el disco de Kaggle |
| `mtp` | `on` | La palanca principal de velocidad |
| `spec` | `4` | Propuestas por paso. `SK_SPEC=2` si ves respuestas raras; `6` es más rápido en código pero a veces más lento en prosa |
| `spec_min_p` | `0.5` | Confianza mínima para proponer; es el valor que Strata usa en sus configs rápidas |
| `mtp_q4` | `all` | Draft en 4 bits: +3.5% de decode y las respuestas no cambian (el modelo verifica) |
| `context` | `32k` | Suficiente para código; menos contexto = más VRAM para expertos |
| `kv` | `int8` | Buena calidad con la mitad de memoria que `fp16` |
| `vision` | `off` | Libera ~0.9 GB de VRAM para expertos (no se usan imágenes aquí) |
| `expert_cache` | `auto` | **Nunca un número fijo**: pasarse de slots puede tirar el decode hasta 7× |
| `vram_reserve_mib` | `512` | Kaggle no tiene escritorio; se reserva lo mínimo |
| `resident_mode` | `budget` | Los expertos que no caben en VRAM se copian a RAM (`SK_RESIDENT=all` usa `--resident-experts`) |
| `resident_reserve_gib` | `14` | RAM que se deja libre para SO, motor y notebook; el motor además nunca fija más de lo que cabe |
| `arena_mmap` | `0` | `SK_ARENA_MMAP=1` usa `STRATA_ARENA_MMAP=1` + `--pcie-frac 0`, el modo que la doc recomienda para 2×16 GB + 32 GB de RAM |
| `pipeline_windows` | `0` | `2` es un experimento NVIDIA que quita VRAM al caché de expertos |
| `pool_workers` | `0` | `0` = auto (un worker por núcleo físico) |
| `bf16_tc` | `0` | `1` acelera el **prompt** 20–34% en Turing, pero cambia el redondeo |
| `tune` | `skip` | El tuning cuesta 20 min y en Kaggle no compensa |
| `access` | `cloudflare` | `SK_ACCESS=local` para una sesión privada sin túnel |
| `show_key` | `false` | `SK_SHOW_KEY=1` imprime la API key completa al conectar |

## Qué significa cada número del benchmark

- **TTFT**: tiempo hasta el primer token. Incluye el prefill del prompt; baja mucho en la segunda pasada.
- **tok/s (decode)**: tokens generados por segundo, medidos desde el primer token hasta el último.
  Es la cifra que importa para saber si MTP está funcionando.
- **mediana de 3**: la primera pasada paga cachés frías; la mediana es la cifra honesta.

Cifras de referencia de la documentación de Strata (en **su** hardware, no en Kaggle, sirven solo para entender el
efecto de cada ajuste):

| Ajuste | Medido en | Efecto |
|---|---|---|
| MTP vs sin MTP | Apple M5 Max, IQ3_S | 87.4 → 114–119.5 tok/s |
| `--mtp-q4 all` | DGX Spark | +3.5% decode, mismas respuestas |
| `--resident-experts` en split | 2 GPUs, 32 GB RAM | 32/64 → 71/69 tok/s |
| Tuning `--tune quick` | — | No hay medición pública de ganancia; por eso se omite |

## Túnel de Cloudflare

Por defecto el notebook arranca con **`access = "cloudflare"`**: el servidor escucha solo en `127.0.0.1` y Strata
levanta un *quick tunnel* de Cloudflare, así que puedes usar el modelo desde tu portátil o desde otra app.

- **La URL es pública.** Strata crea una **API key** automáticamente y esa key es la única barrera. El notebook la
  muestra enmascarada; se revela con `SK_SHOW_KEY=1` o con `strata launch --show-key`.
- **No compartas el output del notebook con la key visible.**
- **Strata no instala `cloudflared`**: la sección 4 lo descarga a `~/.local/bin` una vez por sesión.
- **La URL cambia en cada sesión** y el túnel vive mientras viva la sesión de Kaggle.
- **Para cerrarlo:** sección 12 (`strata launch --stop`), que para el servidor y el túnel.
- **Para una sesión privada:** `SK_ACCESS=local` antes de ejecutar (o `"access": "local"` en `CONFIG`). El benchmark
  y el SVG siempre usan `127.0.0.1`, así que el túnel no afecta a las cifras de velocidad.

Ejemplo de cliente:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<aleatorio>.trycloudflare.com/v1", api_key="sk-...")
client.chat.completions.create(
    model="<modelo>",
    messages=[{"role": "user", "content": "hola"}],
)
```

La URL y el nombre del modelo salen en la sección 8 del notebook.

## Límites de Kaggle (importante)

- **Disco:** `df` enseña ~1 TB pero el límite real es ~95 GB. El pack + draft MTP ronda los 65 GB. El notebook
  aborta solo si quedan menos de 8 GB libres. Si el disco se vuelve de solo lectura, hay que reiniciar la sesión.
- **RAM:** ~31 GB. El Coder necesita ~23 GB de expertos si ninguno cabe en VRAM; con 2 T4 gran parte cabe en GPU,
  y el resto se copia a RAM con presupuesto. El notebook aborta si la RAM disponible baja de 1.5 GB.
- **Tiempo:** las sesiones de Kaggle duran como máximo 12 h (y hay cuota semanal de GPU). La descarga inicial se
  repite en cada sesión.
- **GPU:** si Kaggle te da una sola T4, el modelo sigue arrancando pero será bastante más lento. El preflight avisa.

## Problemas frecuentes

| Síntoma | Causa y arreglo |
|---|---|
| El plan dice **Swift** o **IQ2_XS** | Falta `--family coder`. Este notebook lo pasa siempre; no lo quites |
| La sección 7 marca `FALTA` en algo | Strata rechazó ese ajuste; busca el aviso en `strata.log`. El resto sigue funcionando |
| Decode muy lento (< 2 tok/s) | Los expertos están cayendo al SSD. Revisa RAM libre; prueba `SK_CONTEXT=16k` o `SK_ARENA_MMAP=1` |
| La sesión se congela al arrancar | Presupuesto de RAM demasiado ajustado. Sube `resident_reserve_gib` (menos expertos en RAM) o usa `SK_ARENA_MMAP=1` |
| `No space left on device` | Reinicia la sesión de Kaggle; el disco vuelve limpio |
| La sección 8 no encuentra la URL del túnel | `cloudflared` no está o falló: comprueba `~/.local/bin/cloudflared --version`. El modelo sigue disponible en `127.0.0.1` |
| La URL pública responde 401 | Falta la API key en el cliente: `strata launch --show-key` y mándala en `Authorization: Bearer …` |
| Sección 6: «el proceso arrancó pero `/v1/models` no responde» | El servidor sí está sirviendo, pero `/v1/models` exige key incluso en `127.0.0.1`: el notebook la lee solo con `strata launch --show-key` (o `STRATA_API_KEY`). Comprueba también que el error no sea 401 en `strata.log` |
| El puerto 8080 está ocupado | Ejecuta la sección 12 (`--stop`) o reinicia la sesión. La sección 6 no relanza si ya hay servidor |
| Respuestas peores de lo esperado | Es un pack de ~1 bit con la mitad de expertos: es el precio de que quepa. Para código rinde; fuera de código es más flojo |

## Estructura

```
strataKaggle/
├── strata_kaggle.ipynb     # el notebook (todo el código, autocontenido)
├── docs/OPTIMIZATIONS.md   # por qué cada flag, con citas a la doc de Strata
├── README.md
├── LICENSE
├── .gitignore
└── .gitattributes
```

El notebook es autocontenido a propósito: en Kaggle se sube un solo archivo y no hay que clonar el repo.

## Créditos

- [Strata](https://github.com/Niko1221/Strata) — el motor (MIT).
- [Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) — el modelo, pack Coder de ISTA-DASLab.
- [jigs10/qwen3.8-flash-strata-kaggle](https://github.com/jigs10/qwen3.8-flash-strata-kaggle) — el notebook original.

Este repo solo cambia la configuración y el flujo de trabajo; no modifica ni redistribuye el modelo.
