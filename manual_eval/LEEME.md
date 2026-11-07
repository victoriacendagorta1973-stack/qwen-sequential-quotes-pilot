# Evaluación manual: doce textos, cuatro familias

Comparación del Qwen/Qwen2.5-7B-Instruct oficial, revisión a09a35458c702b33eeacc393d103063234e8bc28,
con el mismo modelo y el adaptador N3 congelado del piloto secuencial del 6 de noviembre.
El paquete incluye los dos archivos exactos del adaptador y comprueba sus SHA256.

Son cuatro familias nuevas escritas manualmente por Victoria Cendagorta Villalba, con tres niveles cada una.
Se copia el último lote completo de este chat, sin reformular, corregir ni normalizar el contenido.
Se excluyen únicamente los separadores de párrafo del mensaje y los espacios finales de formato.
Se conserva literalmente Dodotea y toda la puntuación, aunque el modelo quiera corregirlos.
Python construye el input sustituyendo únicamente los seis delimitadores por comillas dobles rectas.
No se usa texto generado por modelos cerrados como datos. Las referencias están revisadas antes de obtener predicciones.

Este lote se reserva exclusivamente para evaluación: no se entrena, no se ajustan parámetros y no se genera más texto sintético.
Ambos evaluandos reciben la misma instrucción del piloto, sin ejemplos, y solo el input de cada caso.
Inferencia greedy, semilla 42, máximo 1024 tokens nuevos, sin truncamiento ni reparación posterior de respuestas.
La coincidencia exacta exige conservar palabras, espacios y puntuación; se reportan también jerarquía y conservación del contenido.
No se incorporan controles nuevos: sus apartados tendrán n=0. No son puntuaciones cero.
Los resultados son descriptivos de 12 casos relacionados en 4 familias, no 12 muestras independientes.
No se han medido GSM8K, IFEval ni MMLU y esta prueba no demuestra por sí sola la solución del reto de libros.
Las estructuras pueden solaparse parcialmente con las familias del piloto; no se afirma independencia sintáctica completa.

Ejecutar python3 -u ejecutar.py en una GPU CUDA. Dependencias fijadas, comprobación y empaquetado automáticos.
No cambia el adaptador. Máximo global: 20 minutos, más el empaquetado final.
El ZIP de resultados conserva referencias, hashes, código, dependencias, predicciones, métricas y errores si los hubiera.
La ejecución real de los modelos se realiza en el pod, no en la preparación local del paquete.
