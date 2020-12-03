const express = require('express');
const face = require('./facial-recognition');
const app = express();
const port = 3000;
const cors = require('cors');

app.use(express.json());

app.use(cors({ origin: '*', preflightContinue: true }));

app.post('/', async (req, res) => {
  const { picture } = req.body;

  const response = await face.recognizeFace(picture);
  res.send(response);
});

app.listen(port, () => {
  console.log(`Listening at http://localhost:${port}`);
});
