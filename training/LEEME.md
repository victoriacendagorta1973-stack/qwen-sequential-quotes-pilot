# Piloto secuencial N2 → N3

## Alcance y datos

Prueba exploratoria de comillas, sin M1 del cliente ni libros. No demuestra solución al desafío contractual ni conservación de capacidades generales. No se ejecutan GSM8K, IFEval o MMLU y no se afirmará que se conservaron a partir de N0. Los datos, scripts e instrucciones deben auditarse según las reglas del contrato antes de una entrega contractual; este piloto no certifica admisibilidad ni autoría de todo el material.

Se excluye por completo P1, de procedencia mixta. Las seis familias restantes son las aportadas por la usuaria y las sustituciones de Qwen. Se incorporan las correcciones deterministas autorizadas (espacios antes de cierre, cierres de exclamación y artículo un para software/dispositivo). 102 registros de comillas, 72 pares distintos. Duplicados conservados e identificados en los datos, métricas por pares únicos y por registros. No son controles de no intervención.

Entrenamiento: P2, P3, P4 y P6. Desarrollo reservado DEL ENTRENAMIENTO: P5 y P7, familias completas (dos citas exteriores e incisos). Ya se han visto sus resultados durante el diagnóstico; no es un test final independiente y no se usa para elegir checkpoints ni detener por calidad. Se entrena un número fijo de pasos y se informa cada punto.

Once controles de identidad extraídos del corregido del corpus propio, sin comillas ni nombres visibles, revisados por lectura del asistente: seis train y cinco dev. Los documentos fuente de controles no cruzan particiones. Esta selección no certifica anonimización completa del corpus maestro. Entrada y salida son idénticas. N0 mide preservación local del texto, no razonamiento general.

## Configuración fijada

Modelo Qwen/Qwen2.5-7B-Instruct, SHA a09a35458c702b33eeacc393d103063234e8bc28. Pesos originales congelados. Solo LoRA en q_proj/k_proj/v_proj/o_proj de los índices 8..19 (capas 9.ª..20.ª). Se comprueba antes de entrenar que existen exactamente 96 tensores entrenables A/B en los 48 módulos previstos, y que ningún otro parámetro está activo. Embeddings, lm_head, MLP, normalizaciones y demás parámetros base quedan congelados. No se supone que esto garantice conservación del comportamiento.

r=16, alpha=32, dropout=0, lr máximo 2e-4, cosine con 5% warmup, AdamW sin weight decay, clipping=1. Microbatch=1, acumulación=4, 40 pasos por etapa (160 muestras con reemplazo), semilla=42. Precisión BF16 si la GPU la soporta; en otro caso FP16 con GradScaler. Pesos completos, sin cuantización nueva, gradient checkpointing no reentrante, atención SDPA. Secuencia máxima 768, sin truncamiento silencioso. Etiquetas de pérdida únicamente en la respuesta del asistente y sus tokens finales, nunca en el prompt.

La mezcla N2 contiene 16 controles + 43 N1 + 101 N2. N3 contiene 16 controles + 28 N1 + 43 N2 + 73 N3. Las proporciones 20/30/50 se aproximan por redondeo en las 144 muestras con comillas. Se guardan los IDs efectivamente muestreados. Los duplicados del corpus y el muestreo con reemplazo tienen peso explícito: no se cuentan como nuevos ejemplos independientes.

Un único adaptador se crea tras la baseline, se entrena en N2 y se CONTINÚA en N3; no se reinicia desde el modelo base. El optimizador y su calendario se reinician por etapa, no el adaptador. Se guardan checkpoints cada diez pasos. Al terminar, adapter_N3 es el único adaptador final; adapter_N2 queda como punto de comparación, no como selección por pregunta o por nivel.

## Evaluación y trazabilidad

Baseline nueva antes de crear LoRA, post_N2 y post_N3 sobre el mismo lote de train/dev y controles. Instrucción fija sin ejemplos resueltos, plantilla oficial sin modificaciones. Inferencia greedy, repetition_penalty=1, max_new_tokens=1024. Se preserva la salida sin reparaciones ni strip. Se reutiliza la predicción para entradas idénticas dentro de una evaluación; la caché no usa referencias ni niveles y se reinicia en cada etapa. Cada ID recibe su resultado, y el evaluador exige correspondencia exacta.

Reportes con exactitud, jerarquía y conservación del resto del texto, separados por partición y nivel y deduplicados por par. Las jerarquías en este corpus no incluyen apóstrofos léxicos, por lo que el evaluador simplificado no debe aplicarse sin adaptación a un corpus general. La comparación.json guarda los tres puntos. RESUMEN.txt resume las cuentas; no aplica umbrales de éxito inventados. Que baje la loss no prueba generalización ni ausencia de olvido.

El control por grupos no garantiza ausencia universal de contaminación semántica. Hay parentesco entre construcciones y todos los ejemplos actuales son desarrollo. No se muestran ejemplos de P5/P7 en entrenamiento ni en el prompt. Reservar construcciones nuevas después para un estudio ampliado.

## Ejecución autónoma y límites

python3 ejecutar.py hace todo: instala PEFT 0.21.2 con --no-deps (no actualiza torch/transformers), corre cuatro tests de datos, realiza una comprobación de gradientes con un Qwen diminuto y la misma selección de adaptadores, carga pesos oficiales, mide baseline, entrena N2, evalúa, continúa N3 y evalúa. La prueba diminuta usa IDs artificiales solo para verificar código; no se incorpora a los datos del modelo real.

Límite global de 45 minutos para instalación, pruebas y proceso GPU. El proceso se detiene automáticamente al agotarlo. El empaquetado puede añadir unos segundos. Si hay OOM, incompatibilidad, loss/gradiente no finitos, selección de parámetros incorrecta o presupuesto agotado, se detiene sin cambiar la receta y crea igualmente un ZIP con el error, logs y checkpoints disponibles. No sobrescribe resultados previos.

No hay acceso remoto desde este chat para ejecutar directamente en el pod: la usuaria debe subir el paquete y lanzar el único comando. Después puede dejarlo corriendo. El script no detiene ni elimina el pod: terminar el proceso libera su memoria GPU, pero RunPod sigue cobrando mientras el pod siga activo. Descargar el ZIP antes de detener o eliminar almacenamiento efímero.

No se ha probado este entrenador con GPU en el entorno del asistente. Se han validado localmente los tests puros; la compatibilidad PEFT/Transformers, las etiquetas y los gradientes se verifican automáticamente en el pod antes de modificar pesos del modelo real. Las versiones, hashes, configuración y errores quedan documentados.
