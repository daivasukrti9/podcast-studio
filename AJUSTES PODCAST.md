AJUSTES PODCAST

Para llevar tu pipeline de podcast (corriendo en tu entorno de **Google Antigravity / AGY** con Python) de un audio plano a una conversación indistinguible de la realidad —con 4 estilos distintos, 8 voces neutras y micro-reacciones humanas—, tienes que atacar dos capas por separado: **el motor de guionizado (LLM)** y **el motor de síntesis de voz (TTS) + mezcla en Python**.

---

### 1. Arquitectura de Guionizado Dinámico (Los 4 Perfiles)

El error común al generar guiones con IA es pedir "un diálogo natural". El LLM escribe párrafos perfectos y gramaticalmente impecables que nadie dice en la vida real. Para lograr **muletillas, interrupciones, silencios y humor ligero**, debes inyectar un JSON de configuración de estilo (`style_profile`) en el *System Prompt* de tu agente guionista antes de pasarle el texto fuente.

Define estos 4 perfiles en tu código:

1. **Oficina (Colaborativo y ágil):**
* *Dinámica:* Dos colegas revisando un tema frente a un café. Ritmo medio-rápido.
* *Inyecciones obligatorias:* Frases de validación corta (*"Totalmente"*, *"Claro, te entiendo"*), dudas leves al pensar (*"A ver..."*, *"O sea, lo que pasa es que..."*) y aterrizaje práctico a tareas diarias.


2. **Empresarial (Ejecutivo y estratégico):**
* *Dinámica:* Mesa de análisis o entrevista de liderazgo. Seguro, pausado, con autoridad pero cercano.
* *Inyecciones obligatorias:* Pausas reflexivas antes de dar un dato clave, transiciones elegantes (*"Fíjate en ese punto"*, *"Exacto, ahí está la clave"*), cero jerga callejera, humor muy sutil e intelectual.


3. **Streamer (Energético y reactivo):**
* *Dinámica:* Alta energía, ritmo rápido, reacciones espontáneas, estilo charla en vivo.
* *Inyecciones obligatorias:* Exclamaciones naturales (*"¡No te puedo creer!"*, *"Espera, espera, ¿en serio?"*), risas breves entre frases, bromas ligeras rompiendo la cuarta pared y cambios de entonación bruscos.


4. **Relajante (Nocturno / *Late-night*):**
* *Dinámica:* Tono bajo, íntimo, sin prisa, ideal para temas profundos o *soft skills*.
* *Inyecciones obligatorias:* Puntuación cargada de puntos suspensivos (`...`) para forzar al TTS a bajar el ritmo, suspiros textuales, comentarios contemplativos (*"Mmm, qué buen punto..."*, *"Tal cual..."*).



#### Reglas de Prompting para "Micro-reacciones"

En el prompt del guionista (ya sea que uses Gemini, DeepSeek o Qwen), exige que el output sea una lista de objetos JSON con etiquetas explícitas de actuación y ritmo:

```json
[
  {"speaker": "Host_A", "style": "streamer", "text": "O sea, automatizaron todo el cierre contable en dos días..."},
  {"speaker": "Host_B", "style": "streamer", "text": "[risa ligera] Espera, ¿dos días? ¡Yo tardaba dos semanas en eso!"},
  {"speaker": "Host_A", "style": "streamer", "text": "[pausa_corta] Literal. Y lo mejor es que nadie tuvo que tocar un Excel."}
]

```

* **La regla del 15%:** Instruye al modelo para que al menos el 15% de las intervenciones sean **turnos de relleno puro** (*backchanneling*) sin aportar información nueva: *"Mmm"*, *"Sí, claro"*, *"¡Uf!"*, *"Exacto"*, *"¿En serio?"*.

---

### 2. Selección de las 8 Voces Neutras y Hiperrealistas

Para evitar acentos excesivamente marcados (como un porteño muy cerrado o un castellano muy marcado) y mantener un **español latinoamericano neutro/corporativo** pero extremadamente humano, la mejor estrategia en tus entornos (sin GPU dedicada en tu laptop i7 o corriendo ligero en tu servidor Ubuntu) es combinar motores según tu presupuesto:

#### Opción A: Cero Costo Recurrente (`edge-tts` + ajuste prosódico)

Si quieres mantener el pipeline gratuito en Python, **Microsoft Edge TTS** tiene las mejores voces neuronales en español neutro sin necesidad de GPU. Selecciona estas 8 combinando regiones suaves (México, Colombia, Estados Unidos Latino y Costa Rica/Centroamérica, que son las más neutras al oído internacional):

* **Voces Femeninas (4):**
1. `es-MX-DaliaNeural` (Cálida, versátil, excelente para *Oficina* y *Relajante*).
2. `es-CO-SalomeNeural` (Muy clara, expresiva, ideal para *Streamer* o dinámica).
3. `es-US-PalomaNeural` (Español neutro panregional, tono muy profesional para *Empresarial*).
4. `es-CR-MariaNeural` (Suave, pausada, naturalidad muy orgánica).


* **Voces Masculinas (4):**
5. `es-MX-JorgeNeural` (El estándar de oro en naturalidad; funciona en los 4 perfiles).
6. `es-CO-GonzaloNeural` (Tono amigable, excelente para bromas ligeras y *Oficina*).
7. `es-US-AlonsoNeural` (Voz grave, ejecutiva y firme para *Empresarial*).
8. `es-EC-LuisNeural` (Muy neutro, articulación limpia y serena para *Relajante*).

*(Tip técnico: En `edge-tts`, puedes variar el parámetro `rate` (+5% para Streamer, -8% para Relajante) y `pitch` (-2Hz para Empresarial) dinámicamente según el perfil elegido).*

#### Opción B: Máximo Realismo Emocional (API de bajo costo / *OpenRouter / ElevenLabs / OpenAI*)

Si buscas que el TTS interprete **suspiros, risas reales y cambios de emoción** sin sonar robótico:

* **OpenAI `gpt-4o-mini-tts` (o `tts-1-hd`):** Permite pasar instrucciones de actuación en lenguaje natural (ej. *"Habla con tono relajado, sonriendo ligeramente al final"*).
* **ElevenLabs (Multilingual v2 / Flash v2.5):** Sus voces en español neutro (como *Mateo*, *Valentina*, *Camila*, *Diego*) capturan los puntos suspensivos y las risas escritas en el texto (`"jajaja, bueno..."`) de forma casi indistinguible de un humano.

---

### 3. El Secreto de la Mezcla en Python (`pydub` / `ffmpeg`)

Un diálogo con las mejores voces del mundo sonará falso si hay exactamente 1.0 segundo de silencio entre cada frase. En una charla humana, la gente se pisa un poco o responde en milisegundos.

Modifica tu script ensamblador en Python (usando `pydub`) para procesar las etiquetas del JSON:

1. **Silencios Variables (No fijos):**
* Si es un cambio de turno normal: inserta un silencio aleatorio entre `150ms` y `350ms` (`random.randint(150, 350)`).
* Si el guion trae `[pausa_corta]` o `...`: inserta `550ms` a `800ms`.


2. **Solapamiento (*Audio Ducking / Overlap*):**
* Cuando el turno siguiente sea un comentario corto de validación (*"Totalmente"*, *"Mmm, sí"*, *"¡Exacto!"*), **no esperes a que termine el hablante anterior**. Súbelo a la pista haciendo que empiece `-250ms` antes de que acabe la frase anterior (`audio_final.overlay(clip_corto, position=len(audio_final) - 250)`). Ese pequeño cruce es lo que engaña al cerebro humano y le hace percibir una conversación real en vivo.


3. **Respiraciones y Ambiente (Opcional pero brutal):**
* Añade una pista de fondo (*room tone* o ruido de oficina muy sutil a `-42 dB`) durante todo el episodio. Evita el "silencio digital absoluto" entre frases, que es lo que delata a los audios generados por IA.



---

### Siguiente paso en AGY

Para adaptar esto directamente a tu flujo de trabajo en **Antigravity**:

1. Crea un archivo `profiles.json` en tu carpeta del proyecto con los 4 *system prompts* y los mapeos de voces/velocidades.
2. Ajusta el agente guionista para que devuelva exclusivamente el array JSON con las etiquetas de pausa/solapamiento.
3. Actualiza tu script de ensamblaje de audio para leer esas etiquetas y aplicar los `overlays` de `-250ms`.


----------------------------------------------------

Sí, **hace una diferencia abismal**, pero únicamente si configuras tres cosas: la **prosodia textual**, los **parámetros de estabilidad de la API** y el **ensamblaje no lineal**. Si usas ElevenLabs con un guion plano y configuración por defecto, suena como un locutor de comercial muy caro, no como una charla real.

El verdadero salto ocurre cuando combinas la calidad vocal de ElevenLabs con control fino en Python y herramientas *open-source* diseñadas específicamente para audio conversacional.

---

### 1. ¿Por qué ElevenLabs sí marca la diferencia (y cómo configurarlo bien)?

Los modelos **Eleven Multilingual v2** y **Eleven Flash v2.5** son los únicos que realmente interpretan la intención emocional oculta en la puntuación y en las palabras de relleno sin necesidad de etiquetas SSML complejas.

Para que las voces no suenen "perfectas de estudio" sino "humanas y espontáneas", ajusta estos parámetros en cada llamada a su API:

* **`stability` (Estabilidad) entre `0.30` y `0.45`:** Por defecto viene en `0.50` o más, lo que aplana la emoción. Bajarla a `0.35` obliga al modelo a introducir micro-inflexiones, dudas y cambios de tono naturales en cada frase. (Por debajo de `0.25` puede empezar a sonar errático).
* **`similarity_boost` entre `0.75` y `0.85`:** Mantiene la identidad de la voz neutra sin deformarla cuando el tono sube en el perfil *Streamer*.
* **`style` (Exageración de estilo) entre `0.15` y `0.35`:** Dale `0.35` para *Streamer* (más expresivo y teatral) y `0.0` o `0.10` para *Empresarial* y *Relajante*.
* **El truco del contexto (`previous_text` y `next_text`):** La API de ElevenLabs tiene dos parámetros que casi nadie usa en pipelines automatizados: `previous_text` y `next_text`. Si le pasas la frase del hablante anterior y la siguiente en cada request, ElevenLabs ajusta automáticamente la entonación de entrada y salida para que parezca que **realmente está respondiendo** a lo que acaba de escuchar.
* **Escritura fonética de emociones:** ElevenLabs lee las risas y dudas si las escribes como suenan: `"Ja, ja, ja, bueno... a ver, déjame pensar..."`, `"¡Uff! Qué tema..."`, `"Mmm, no estoy tan seguro de eso"`.

*(Nota de arquitectura: Como generas episodios de ~20 minutos, usa **Flash v2.5** para borradores o episodios internos diarios —cuesta la mitad de créditos y genera en milisegundos— y reserva **Multilingual v2** para los episodios insignia).*

---

### 2. Recursos en GitHub que aportan valor real a tu pipeline

Para no reinventar la rueda en tu entorno de **Antigravity (AGY)** y conectar esto directo con Python, estos repositorios te resuelven el guionizado multi-agente, la diarización natural y la mezcla de audio:

1. **`souzatharsis/podcastfy`**
* **Qué aporta:** Es el estándar *open-source* para convertir documentos/URLs en conversaciones estilo NotebookLM usando Python.
* **Por qué te sirve:** Tiene soporte nativo para múltiples backends de TTS (incluyendo ElevenLabs, Edge-TTS y modelos locales) y su arquitectura de generación de guiones permite inyectar *custom conversation styles* (ahí puedes meter tus 4 perfiles: Oficina, Empresarial, Streamer, Relajante) y roles de hablantes sin programar el orquestador desde cero.


2. **`lfnovo/podcast-creator` (y `open-notebook`)**
* **Qué aporta:** Un paquete Python modular pensado para pipelines automatizados que separa estrictamente la generación de la estructura narrativa de la síntesis de voz.
* **Por qué te sirve:** Permite definir perfiles de hablantes con personalidades distintas en archivos de configuración limpios (YAML/JSON) y maneja el ensamblaje de pistas con transiciones mucho más orgánicas que una concatenación simple.


3. **`yujxx/podagent`**
* **Qué aporta:** Un enfoque multi-agente para producción de podcasts donde un agente redacta, otro critica el ritmo y otro ajusta las etiquetas de audio.
* **Por qué te sirve:** Puedes tomar su patrón de "agente crítico" para que revise el guion generado y le inyecte las bromas ligeras, los silencios cortos y las interrupciones antes de enviarlo a ElevenLabs.


4. **`SWivid/F5-TTS` o `hexgrad/kokoro` (Alternativa híbrida local/cero costo)**
* **Qué aporta:** Si en algún momento quieres clonar o diseñar tus 8 voces propias en tu servidor Ubuntu o con APIs sin pagar por minuto en ElevenLabs, **F5-TTS** tiene una capacidad sobresaliente para capturar ritmo conversacional y pausas naturales usando solo 5-10 segundos de audio de referencia por voz.



---

### 3. Cómo unirlo todo en tu flujo de AGY

La combinación ganadora para tu caso es:

1. **Agente Guionista (LLM):** Genera el JSON estructurado usando la estructura de prompts de *Podcastfy* modificada con tus 4 estilos y la regla de *backchanneling* (comentarios cortos sin contenido nuevo).
2. **Sintetizador (ElevenLabs API):** Envía cada turno pasando `previous_text` y `next_text` + `stability=0.35`. Selecciona 8 voces de la *Voice Library* de ElevenLabs filtrando por idioma `Spanish` y acento `Latin American` (evitando etiquetas regionales fuertes) para garantizar esa neutralidad internacional.
3. **Mezclador (`pydub`):** Aplica el solapamiento de `-250ms` en las respuestas cortas (*"Totalmente"*, *"Ja, sí"*) y añade una pista base de *room tone* sutil para eliminar el vacío digital.



