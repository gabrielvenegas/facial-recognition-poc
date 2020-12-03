require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const path = require('path');
const fetch = require('node-fetch');
const { Sequelize } = require('sequelize');
const redis = require('redis');
const client = redis.createClient();
const { promisify } = require('util');

client.on('error', function (error) {
  console.error(error);
});

const init = async () => {
  faceapi.env.monkeyPatch({ fetch: fetch });
  const MODELS_URL = path.join(__dirname, '/models');

  // Load the face detection models
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODELS_URL);

  // Load the face landmark models
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_URL);

  // Load the face recognition models
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_URL);

  // make image canvas
  const { Canvas, Image } = canvas;
  faceapi.env.monkeyPatch({ Canvas, Image });
};

const recognizeFace = async (picture) => {
  const allDistances = [];
  const users = await findUser();

  // calculate descriptors from image coming from the request
  const img = await canvas.loadImage(picture);
  const { descriptor: descriptor1 } = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  for (let i of users) {
    const { picture, name, mail, id } = i;
    const getAsync = promisify(client.get).bind(client);
    const cachedUsers = JSON.parse(await getAsync('cached-users'));

    // is user cached
    const userIsCached = !!cachedUsers
      ? !!cachedUsers.find(({ id: cachedId }) => cachedId === id)
      : false;

    if (!userIsCached) {
      console.log('NOT CACHED USER');

      const img2 = await canvas.loadImage(picture);

      const { descriptor: descriptor2 } = await faceapi
        .detectSingleFace(img2)
        .withFaceLandmarks()
        .withFaceDescriptor();

      // convert for saving purpose
      const descriptorText = JSON.stringify(Array.from(descriptor2));

      // get distance
      const distance = faceapi.euclideanDistance(descriptor1, descriptor2);

      // push to last result array
      allDistances.push({
        id,
        name,
        mail,
        descriptors: descriptorText,
        distance,
      });
    } else {
      console.log('CACHED USER');

      const idx = cachedUsers.findIndex(({ id: cachedId }) => cachedId === id);
      const userCached = cachedUsers[idx];

      // convert back to float 32
      const descriptorFloat32 = new Float32Array(
        JSON.parse(userCached.descriptors)
      );

      const distance = faceapi.euclideanDistance(
        descriptor1,
        descriptorFloat32
      );

      allDistances.push({
        id,
        name,
        mail,
        descriptors: userCached.descriptors,
        distance,
      });
    }
  }

  // TODO - only cache when necessary
  // cache users and its descriptors
  const usersToCache = JSON.stringify(allDistances);
  client.set('cached-users', usersToCache);

  // get best corresponding
  const min = Math.min(...allDistances.map(({ distance }) => distance));

  return allDistances
    .filter(({ distance }) => distance === min)
    .map(({ id, name, mail, distance }) => ({ id, name, mail, distance }));
};

const findUser = async () => {
  // start sequelize
  const sequelize = new Sequelize('wine-poc', 'foo', 'bar', {
    host: 'localhost',
    dialect: 'mysql',
    port: 3306,
  });

  // fetch users pictures
  const users = await sequelize.query('select * from users');

  sequelize.close();
  return users[0];
};

init();

module.exports = { recognizeFace };
