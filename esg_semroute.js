import { pipeline, cos_sim } from '@xenova/transformers';
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

// LOAD CSV
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


const ENV_SAMPLES = 20;
const NON_ENV_SAMPLES = 80;

const selectedEntries = entries.reduce((acc, entry) => {
    if (acc.trueEnvironmental.length < ENV_SAMPLES && entry.environmental) {
        acc.trueEnvironmental.push(entry.text);
    } else if (acc.falseEnvironmental.length < NON_ENV_SAMPLES && !entry.environmental) {
        acc.falseEnvironmental.push(entry.text);
    } else {
        acc.testEntries.push(entry);
    }
    return acc;
}, { trueEnvironmental: [], falseEnvironmental: [], testEntries: [] });

const trueEnvironmentalEntries = selectedEntries.trueEnvironmental;
const falseEnvironmentalEntries = selectedEntries.falseEnvironmental;
const testEntries = selectedEntries.testEntries;

const ai = new AI({
  //embeddingModel: 'jinaai/jina-embeddings-v2-base-de' // seems amazing, still pretty fast
  embeddingModel: 'Xenova/all-MiniLM-L6-v2' // pretty good and very fast but only english
  //embeddingModel: 'nomic-ai/nomic-embed-text-v1-unsupervised' // pretty okay
});

async function createEmbeddings(samples){
  const embeddingsArray = await ai.embeddings(samples);
  return samples.reduce((s,e,i) => { s[e] = embeddingsArray[i]; return s }, {});
}

const routes = [
    {name: 'environmental', embeddings: await createEmbeddings(trueEnvironmentalEntries)},
    {name: 'not environmental', embeddings: await createEmbeddings(falseEnvironmentalEntries)}
];

async function route(input){ //winner takes all
  const inputEmbedding = (await ai.embeddings([input]))[0];
  let bestRoute = null, bestSimilarity = -1, bestEmbedding = null;
  for(const r of routes){
      for(const [text, embedding] of Object.entries(r.embeddings)){
          const similarity = cos_sim(inputEmbedding, embedding);
          if(similarity > bestSimilarity){
              bestRoute = r.name;
              bestSimilarity = similarity;
          }
      }
  }
  return {input, route: bestRoute, score: bestSimilarity}
}


const stats = {correct: 0, incorrect: 0};
await ai.embeddings(['preload']) // lazy load model to get correct timing after
const st = Date.now();
for(const entry of testEntries){
  const expected = entry.environmental ? 'environmental' : 'not environmental';
  const actual = (await route(entry.text)).route;
  stats[expected == actual ? 'correct' : 'incorrect']++;
}

console.log(`Time: ${Date.now() - st}ms`);
console.log(stats, ((stats.correct / (stats.correct + stats.incorrect))*100).toFixed(2), '% success')
