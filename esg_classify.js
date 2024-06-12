import { pipeline } from '@xenova/transformers';
import fs from 'fs';

class AI {
    constructor({
        embeddingModel='Xenova/all-MiniLM-L6-v2',
        classifierModel='Xenova/nli-deberta-v3-xsmall',
        progressCallback=(...p)=>p.map(e=>console.log(Object.entries(e).map(([k,v])=>`${k}: ${v}`).join(", "))),
    }={}){
        this.embeddingModel = embeddingModel;
        this.embeddingPipeline = null;
        this.classifierModel = classifierModel;
        this.classifierPipeline = null;
        this.progressCallback = progressCallback;
    }

    async embeddings(sentences){
        this.embeddingPipeline ||= await pipeline('feature-extraction', this.embeddingModel, {
            progress_callback: this.progressCallback
        });
        const result = await this.embeddingPipeline(sentences, { pooling: 'mean', normalize: true });
        return result.tolist();
    }

    async classify(text, labels, {multi_label=true}={}){
        this.classifierPipeline ||= await pipeline('zero-shot-classification', this.classifierModel, {
            progress_callback: this.progressCallback
        });
        const result = await this.classifierPipeline(text, labels, { multi_label });
        return result.labels.reduce((acc, label, index) => {
            acc[label] = result.scores[index];
            return acc;
        }, {});
    }
}

// READ CSV
// 0 ist "nicht environment", 1 ist "ist environment"
const data = fs.readFileSync('environmental_2k.csv', 'utf8');
const results = [];
const lines = data.split('\n');
const headers = lines[0].split(',');
for (let i = 1; i < lines.length; i++) {
  const line = lines[i];
  if (line.trim() === '') continue; // Skip empty lines
  const values = [];
  let current = '';
  let inQuotes = false;
  for (let char of line) {
    if (char === '"' && !inQuotes) {
      inQuotes = true;
    } else if (char === '"' && inQuotes) {
      inQuotes = false;
    } else if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  values.push(current); // Add the last value
  const entry = {};
  for (let j = 0; j < headers.length; j++) {
    entry[headers[j]] = values[j];
  }
  results.push(entry);
}
const entries = results.map(result => ({
    text: result.text,
    environmental: result.env === "1"
})).filter(entry => entry.text !== undefined).sort(() => Math.random() - 0.5);

const environmentalCount = entries.filter(entry => entry.environmental).length;
const nonEnvironmentalCount = entries.length - environmentalCount;
console.log(`Number of environmental entries: ${environmentalCount}`);
console.log(`Number of non-environmental entries: ${nonEnvironmentalCount}`);

const ai = new AI();
const ENV_LBL = 'environmental';
const NON_LBL = 'random other';
await ai.classify('Foobar', [ENV_LBL, NON_LBL]); // lazy load the model
const stats = {correct: 0, incorrect: 0};
const st = Date.now();
for (const entry of entries) {
  const result = await ai.classify(entry.text, [ENV_LBL, NON_LBL], { multi_label: false });
  const predictedLabel = result[ENV_LBL] > result[NON_LBL] ? ENV_LBL : NON_LBL;
  const actualLabel = entry.environmental ? ENV_LBL : NON_LBL;
  if (predictedLabel === actualLabel) {
    stats.correct++;
  } else {
    stats.incorrect++;
  }
}
console.log(`Time: ${Date.now() - st}ms`);
console.log(stats, ((stats.correct / (stats.correct + stats.incorrect))*100).toFixed(2), '% success')

