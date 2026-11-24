# Por qué cada ajuste

Este documento justifica cada decisión del notebook con la documentación de Strata. Las citas son
`archivo:línea` del repositorio [Niko1221/Strata](https://github.com/Niko1221/Strata) en `main`
(versión de docs 0.2.20, engine 0.1.66). Las cifras son de **su** hardware: no son mediciones de Kaggle.

---

## El problema real: no cabe en VRAM

| Recurso de Kaggle (T4 x2) | Cantidad |
|---|---|
| VRAM | 2 × 16 GB = **32 GB** |
| RAM | **~31 GB** |
| Disco útil | **~95 GB** |
| Modelo (Coder IQ1_M) | **~58 GB** + ~6 GB de draft MTP |

El modelo no entra en VRAM. Strata lo parte: capas y expertos en las dos GPU, el resto en RAM, y lo que no cabe en
ninguno de los dos se lee del SSD **en cada token**:

> "An NVMe SSD matters, because experts that fit neither VRAM nor the RAM budget are read from it for every token."
> — `docs/UNSLOTH_Q4.md:81`

De ahí salen las dos palancas grandes:

1. **Reducir el número de lecturas de pesos por token** → MTP / speculative decoding.
2. **Evitar que los expertos caigan al SSD** → modo residente en RAM.

---

## Ajustes aplicados

### 1. MTP encendido

```sh
--mtp on --drafting mtp --set spec=4 --set spec-min-p=0.5 --set mtp-q4=all
```

| Hecho | Fuente |
|---|---|
| `--mtp` viene **encendido por defecto**; `off` ahorra ~1 GB de VRAM y es más lento | `docs/TERMINAL_LAUNCH.md:97` |
| `--spec N` = propuestas por paso; las configs rápidas de Strata usan `--spec 4 --spec-min-p 0.5` | `docs/UNSLOTH_Q4.md:139-140`, `docs/AMD_HIP.md:827`, `docs/GYRO.md:231` |
| `--mtp-q4 all` = draft en 4 bits: **+3.5% decode** y las respuestas no cambian | `docs/DGX_SPARK_LINUX.md:44` |
| MTP no cambia las respuestas: solo propone y el modelo completo verifica | `docs/MAC.md:174` |
| `--spec 4` frente a `--spec 2`/`8`: 4 es el punto dulce documentado; 6+ gana en código y pierde en prosa | `docs/INTEL.md:235`, `docs/DGX_SPARK_LINUX.md` |

**Por qué `--drafting mtp` y no `auto`:** `auto` también elige MTP hoy, pero `dflash2` existe para
Qwen3.8-Flash-Next y descarga otro drafter. DFlash2 gana en código greedy (+5–14%) pero **pierde en texto muestreado
y en turnos de agente**, y fuerza una sola petición simultánea (`docs/SPLIT_ENGINE.md:85`, `docs/TERMINAL_LAUNCH.md:96`).
El benchmark del notebook usa `temperature=0`, pero el uso real (y la celda SVG) muestrea; MTP es la apuesta segura.
Se puede probar `--drafting dflash2` cambiando `CONFIG["drafting"]`.

### 2. Expertos residentes en RAM

```sh
--set resident-budget-gib=<RAM total − 14>
```

| Hecho | Fuente |
|---|---|
| `--resident-experts`: copia a RAM exactamente los expertos que la GPU no tiene; **nada se lee del SSD mientras responde** | `docs/DETAILS.md:267` |
| `--resident-budget-gib N`: copia los N GiB más "calientes" según el perfil, implica `--mmap-experts` y deja 4 GiB de margen | `docs/DETAILS.md:268` |
| El motor **nunca fija más de lo que cabe**: lee la RAM libre al arrancar y baja el presupuesto si no entra | `docs/UNSLOTH_Q4.md:176-182` |
| En Linux el page-lock va por pasos de 1 GiB y deja el resto paginable con un aviso, en vez de morir por OOM (#1250, desde 0.1.40.2) | `docs/TROUBLESHOOTING.md:271-279` |
| El modo residente **funciona en layer split** desde el engine 0.1.40 | `docs/MULTI_GPU.md:158`, `terminal/setup.py:6903` |
| Medición en split: `--mmap-experts` 32/64 tok/s → `--resident-experts` (22 GiB en RAM) **71/69 tok/s** | `docs/MULTI_GPU.md:296-303` |
| El Coder IQ1_M necesita ~23 GB de RAM para todos sus expertos | `docs/MODELS.md:74` |

En Kaggle se usa **presupuesto** por defecto: deja 14 GiB para SO, motor y notebook, y el propio motor lo recorta si
no cabe. Si el modelo pidiera más, lee el resto del disco en lugar de provocar un OOM que mataría la sesión.
Con `SK_RESIDENT=all` se activa `--resident-experts` (algo más rápido si todo cabe; el page-lock tiene su propia guarda).

**Alternativa documentada para este hardware exacto.** Strata recomienda, para
*"2x 16 GB GPUs with 32 GB of RAM"* (literalmente Kaggle), mapear la arena del pack nativo en vez de copiarla:

```sh
--set STRATA_ARENA_MMAP=1 --set pcie-frac=0
```

> "maps a native pack's expert arena read-only from `experts.bin` instead of reading it into locked RAM
> (for example 2x 16 GB GPUs with 32 GB of RAM). Run it with `--pcie-frac 0`."
> — `docs/DETAILS.md:288-289`

Se activa con `SK_ARENA_MMAP=1`. Es el modo más robusto contra el OOM; el presupuesto con perfil suele ser más rápido
porque elige los expertos por frecuencia de uso y no por LRU del sistema. El notebook permite comparar ambos.

**Si la sesión se congela** (el caso #649: el presupuesto se fija en RAM y deja poco al sistema), las salidas
documentadas son subir `resident_reserve_gib` (menos presupuesto) o `STRATA_RESIDENT_PIN=0`, que lo deja paginable
a cambio de que la CPU calcule cada fallo (`docs/TROUBLESHOOTING.md:259-261`, `docs/UNSLOTH_Q4.md:184-187`).

### 3. Expert cache automático

```sh
--set expert-cache=auto
```

| Hecho | Fuente |
|---|---|
| Un `--expert-cache N` demasiado grande puede hundir el decode: **de 100 a 14 tok/s** con unos cientos de slots de más | `docs/TROUBLESHOOTING.md:254-257` |
| En una GPU pequeña, "most experts come from the SSD and it is [slow]" | `docs/MODELS.md:82-83` |

Nunca se fija un número: `auto` reparte según la VRAM libre real de cada tarjeta.

### 4. Contexto y KV

```sh
--context 32k --kv int8
```

| Hecho | Fuente |
|---|---|
| `--kv int8` es el formato por defecto; `q4_0` usa la mitad de memoria | `docs/TERMINAL_LAUNCH.md:94` |
| Menos contexto = menos KV = más VRAM para el caché de expertos | `docs/TERMINAL_LAUNCH.md:50` |

32k es suficiente para código y prompts de agente. Si necesitas más, cada salto cuesta VRAM que se le quita a los
expertos; si vas justo, `SK_CONTEXT=16k` suele ser más rápido.

### 5. Visión apagada

```sh
--vision off
```

| Hecho | Fuente |
|---|---|
| El lector de imágenes añade ~0.9 GB | `docs/TERMINAL_LAUNCH.md:51-53` |

En Kaggle no se usan imágenes. Esos 0.9 GB van al caché de expertos.

### 6. Reserva de VRAM mínima

```sh
--set vram-reserve-mib=512
```

Kaggle no tiene escritorio ni navegador consumiendo VRAM, así que la reserva por defecto (pensada para un PC con
pantalla) es dinero tirado. 512 MiB deja margen para los kernels y el draft.

### 7. Prefill automático

```sh
--set prefill=auto
```

| Hecho | Fuente |
|---|---|
| `--prefill auto` elige el trozo según la máquina; es el valor que escriben las configs de Strata | `docs/AMD_HIP_PERFORMANCE.md:32`, `docs/UNSLOTH_Q4.md:139` |
| Trozos fijos (`2048`) solo ganan en prompts largos y pierden en cortos | `docs/MULTI_GPU.md:307-312` |

El benchmark mide TTFT; si tus prompts son largos y el TTFT duele, prueba `--set prefill=8192`.

### 8. Tuning omitido

```sh
--tune skip
```

| Hecho | Fuente |
|---|---|
| El tuning "quick" tarda ~20 min (hasta 2 h el thorough) | `docs/TERMINAL_LAUNCH.md:57` |
| Por línea de comandos **no se ejecuta** salvo que añadas `--tune` | `docs/TERMINAL_LAUNCH.md:57` |

En Kaggle, 20 minutos de una sesión de 12 h para una mejora que no está publicada como significativa no compensa.
Si quieres probarlo: `CONFIG["tune"] = "quick"`.

---

## Lo que se dejó fuera (a propósito)

| Opción | Por qué no |
|---|---|
| `STRATA_BF16_TC=1` | En Turing acelera el **prompt** 20–34%, pero **no el decode** y cambia el redondeo (3 de 9 respuestas distintas). Disponible con `SK_BF16_TC=1` (`docs/OLDER_GPUS.md:296-298`) |
| `--pipeline-windows 2` | Opt-in en NVIDIA; cuesta **160 MiB de caché de expertos por tarjeta** y una segunda ventana de verificación. En 32 GB de VRAM el caché es oro. Disponible con `SK_PIPELINE_WINDOWS=2` (`docs/MULTI_GPU.md:182-198`) |
| `--parallel N` | Para una sola conversación no ayuda; con MTP el camino más rápido es un stream (`docs/BATCHING.md:168`) |
| `--lookup-chain` / `--lookup-max-t` | Pensado para reescrituras de archivos; en layer split queda limitado a 8 (`docs/DETAILS.md:1255`) |
| `--drafting dflash2` | Ver arriba: pierde en texto muestreado y obliga a 1 petición |
| `--expert-cache N` | Ver arriba: puede tirar el decode 7× |
| `--split-skip-if-fits` | Solo tiene sentido si una tarjeta lo puede todo; aquí no |
| `--resident-experts` por defecto | Se prefiere el presupuesto (más acotado); disponible con `SK_RESIDENT=all` |
| `STRATA_ARENA_MMAP=1` por defecto | Es el modo documentado para 2×16 GB + 32 GB, pero pierde la selección de expertos por frecuencia; disponible con `SK_ARENA_MMAP=1` |

---

## Detalle del comando

El notebook construye **un único** array de argumentos (`launch_argv`) y lo usa tanto para `--plan` como para el
arranque, así que lo que validas es exactamente lo que se ejecuta:

```
strata launch --family coder --model iq1_m --context 32k --kv int8 --vision off \
  --mtp on --drafting mtp --tune skip --access cloudflare --port 8080 \
  --set spec=4 --set spec-min-p=0.5 --set mtp-q4=all --set prefill=auto \
  --set expert-cache=auto --set vram-reserve-mib=512 \
  --set resident-budget-gib=<RAM−14> --yes --no-tui
```

Sobre `--set`: el valor se traduce a un flag del motor. Un booleano se escribe **sin valor o con `=on`**; `=off`
elimina la opción. Por eso el notebook usa `resident-experts=on` y no `=1`
(`terminal/strata_cli/launch/settings.py:311-336`).

Sobre `--no-tui`: descarga, sirve, imprime la dirección y **devuelve el control**; el servidor sigue en segundo plano
(`AGENTS.md:46-49`). Eso elimina el `Popen` + `sleep` del original y hace que re-ejecutar la celda sea seguro.

### Túnel de Cloudflare

`--access cloudflare` **no cambia el decode**: el servidor sigue escuchando en `127.0.0.1` y `cloudflared` solo
reenvía tráfico. El benchmark y el SVG van por localhost a propósito, así que las cifras no incluyen la latencia
del túnel.

| Hecho | Fuente |
|---|---|
| Strata **no** descarga `cloudflared`; lo busca en el PATH y en las rutas habituales | `docs/SAFETY.md:74`, `terminal/strata_cli/launch/access.py:122-188` |
| Con un acceso público, Strata crea una **API key** automáticamente (salvo `--no-api-key`, que avisa) | `docs/SAFETY.md:52`, `docs/SAFETY.md:90-92` |
| `start_tunnel` lanza `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:<puerto>`, espera hasta 45 s y guarda la URL en `<nombre>.tunnel.json` | `terminal/strata_cli/launch/access.py:438-475` |
| `strata launch --show-key` imprime la key real; sin el flag, el launcher la muestra enmascarada | `terminal/strata_cli/launch/cmd.py:798-802`, `docs/TERMINAL_LAUNCH.md:113` |

---

## Cómo comprobar que se aplicó

La sección 7 lee el run config que escribe el launcher (`…/launch/<nombre>.json`) y compara `args` con lo pedido.
Es la única forma fiable: un flag en la línea de comandos no garantiza que el motor lo acepte, y Strata avisa y sigue
en vez de negarse.

Si algo sale `FALTA`, el aviso correspondiente está en `strata.log`.

---

## Referencias

- `docs/TERMINAL_LAUNCH.md` — flags de `strata launch`, valores por defecto.
- `docs/MULTI_GPU.md` — layer split, modo residente, pipeline windows.
- `docs/MODELS.md` — tamaños, RAM necesaria, Coder.
- `docs/DETAILS.md` — semántica de `--resident-experts`, `--resident-budget-gib`, `--spec`, `--mtp-q4`.
- `docs/OLDER_GPUS.md` — switches de Turing (sm_75).
- `docs/TROUBLESHOOTING.md` — `--expert-cache` y OOM.
- `docs/MAC.md` — MTP no altera las respuestas.
