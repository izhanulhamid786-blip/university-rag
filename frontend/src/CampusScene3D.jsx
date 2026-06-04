import { useEffect, useRef } from "react";
import * as THREE from "three";

const LOGO_TEXTURE = "/CUKLogo.png";

function makeParticleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
  gradient.addColorStop(0, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.45, "rgba(255,197,71,0.72)");
  gradient.addColorStop(1, "rgba(255,197,71,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

export default function CampusScene3D() {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 0.35, 8);

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.setAttribute("aria-hidden", "true");
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 1.8);
    scene.add(ambient);

    const keyLight = new THREE.DirectionalLight(0xfff1c2, 2.2);
    keyLight.position.set(2.4, 3.2, 4);
    scene.add(keyLight);

    const logoGroup = new THREE.Group();
    const logoTexture = new THREE.TextureLoader().load(LOGO_TEXTURE);
    logoTexture.colorSpace = THREE.SRGBColorSpace;
    const logoMaterial = new THREE.MeshBasicMaterial({
      map: logoTexture,
      transparent: true,
      side: THREE.DoubleSide,
    });
    const logo = new THREE.Mesh(new THREE.PlaneGeometry(2.1, 2.1), logoMaterial);
    logo.position.set(0, 0.75, 0);
    logoGroup.add(logo);

    const ringMaterial = new THREE.MeshStandardMaterial({
      color: 0x225d49,
      emissive: 0x123b30,
      emissiveIntensity: 0.24,
      metalness: 0.22,
      roughness: 0.32,
      transparent: true,
      opacity: 0.7,
    });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.25, 0.025, 16, 128), ringMaterial);
    ring.position.copy(logo.position);
    ring.rotation.x = Math.PI / 2.9;
    logoGroup.add(ring);
    scene.add(logoGroup);

    const particleCount = 90;
    const positions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particleCount; index += 1) {
      positions[index * 3] = (Math.random() - 0.5) * 9;
      positions[index * 3 + 1] = (Math.random() - 0.5) * 5.2;
      positions[index * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const particles = new THREE.Points(
      particleGeometry,
      new THREE.PointsMaterial({
        map: makeParticleTexture(),
        size: 0.08,
        transparent: true,
        depthWrite: false,
        opacity: 0.62,
        blending: THREE.AdditiveBlending,
      }),
    );
    scene.add(particles);

    const pointer = { x: 0, y: 0 };
    const logoBase = { x: 0, y: 0.75, scale: 1 };
    const handlePointerMove = (event) => {
      const rect = mount.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
      pointer.y = ((event.clientY - rect.top) / rect.height - 0.5) * -2;
    };
    window.addEventListener("pointermove", handlePointerMove);

    let frameId = 0;
    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);

      const narrow = width / height < 0.75;
      logoBase.x = 0;
      logoBase.y = narrow ? -0.08 : 0.75;
      logoBase.scale = narrow ? 0.62 : 1;
      logo.scale.setScalar(logoBase.scale);
      ring.scale.setScalar(logoBase.scale);
      logo.position.x = logoBase.x;
      ring.position.x = logoBase.x;
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const animate = () => {
      const elapsed = performance.now() * 0.001;
      logoGroup.rotation.y += (pointer.x * 0.18 - logoGroup.rotation.y) * 0.04;
      logoGroup.rotation.x += (pointer.y * 0.08 - logoGroup.rotation.x) * 0.04;
      logo.position.y = logoBase.y + Math.sin(elapsed * 1.1) * 0.08;
      ring.position.y = logo.position.y;
      ring.rotation.z = elapsed * 0.35;
      particles.rotation.y = elapsed * 0.035;
      particles.rotation.x = pointer.y * 0.025;
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      window.removeEventListener("pointermove", handlePointerMove);
      logoGeometryDispose(logo);
      ring.geometry.dispose();
      ringMaterial.dispose();
      particleGeometry.dispose();
      particles.material.map?.dispose();
      particles.material.dispose();
      logoTexture.dispose();
      logoMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div className="campus-scene-3d" ref={mountRef} aria-hidden="true" />;
}

function logoGeometryDispose(mesh) {
  mesh.geometry.dispose();
}
