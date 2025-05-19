const bandas = [
  { nombre: "Nirvana", genero: "rock" },
  { nombre: "Nine Inch Nails", genero: "industrial" },
  { nombre: "Backstreet Boys", genero: "pop" },
  { nombre: "N Sync", genero: "pop" },
  { nombre: "Night Club", genero: "techno" },
  { nombre: "Apashe", genero: "techno" },
  { nombre: "STP", genero: "rock" }
];

const generos = ["rock", "pop", "industrial", "techno"];
const generoMap = {
  rock: 0,
  pop: 1,
  industrial: 2,
  techno: 3
};

// Mostrar la lista de bandas para calificar
const contenedor = document.getElementById("bands");
bandas.forEach((banda, i) => {
  const div = document.createElement("div");
  div.className = "band";
  div.innerHTML = `
    <label>${banda.nombre}</label>
    <input type="number" min="1" max="10" id="rate-${i}" value="5">
  `;
  contenedor.appendChild(div);
});

async function procesar() {
  const ratings = bandas.map((_, i) =>
    parseFloat(document.getElementById(`rate-${i}`).value)
  );

  const xs = tf.tensor2d(bandas.map(b => [generoMap[b.genero]]));
  const ys = tf.tensor2d(ratings.map(r => [r]));

  const model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: generos.length,
    outputDim: 2,
    inputLength: 1
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 8, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  await model.fit(xs, ys, {
    epochs: 200,
    verbose: 0
  });

  const predictions = await model.predict(xs).array();

  const recomendadas = bandas.map((b, i) => ({
    nombre: b.nombre,
    score: predictions[i][0]
  }));

  recomendadas.sort((a, b) => b.score - a.score);

  const resultDiv = document.getElementById("resultado");
  resultDiv.innerHTML = "<h2>Ranking de bandas según tus gustos:</h2><ol>" +
    recomendadas.map(b => `<li>${b.nombre} (predicción: ${b.score.toFixed(2)})</li>`).join("") +
    "</ol>";
}
