// <!-- include the Three.js library -->
// <script src="https://cdn.jsdelivr.net/npm/three@0.119.1/build/three.min.js">
    
// </script>

// <!-- create a canvas element to render the 3D scene -->
// <canvas id="canvas"></canvas>

// <!-- load the 3D points from the JSON file -->
// <script>
//   fetch('points.json')
//     .then(response => response.json())
//     .then(points => {
//       // set up the 3D scene
//       const scene = new THREE.Scene();
//       const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
//       const renderer = new THREE.WebGLRenderer({ canvas: canvas });
//       renderer.setSize(window.innerWidth, window.innerHeight);
//       document.body.appendChild(renderer.domElement);

//       // add the points to the scene as 3D objects
//       const geometry = new THREE.Geometry();
//       for (const point of points) {
//         geometry.vertices.push(new THREE.Vector3(point[0], point[1], point[2]));
//       }
//       const material = new THREE.PointsMaterial({ color: 0xff0000 });
//       const pointsObject = new THREE.Points(geometry, material);
//       scene.add(pointsObject);

//       // render the scene
//       function animate() {
//         requestAnimationFrame(animate);
//         renderer.render(scene, camera);
//       }
//       animate();
//     });
// </script>
