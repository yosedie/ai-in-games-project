use bevy::prelude::*;
use rand::Rng;

// Konstanta untuk mempermudah penyesuaian
const PLAYER_SPEED: f32 = 5.0;
const DESIRED_SEPARATION: f32 = 2.0; // Jarak minimal antar NPC

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                player_movement_system,
                // Sistem-sistem ini akan menghitung gaya kemudi (steering force)
                // dan langsung menerapkannya ke Velocity.
                // .chain() memastikan mereka berjalan dalam urutan ini setiap frame.
                (
                    seek_system,
                    flee_system,
                    arrive_system,
                    wander_system,
                    pursuit_system,
                    evade_system,
                    separation_system,
                    containment_system,
                )
                    .chain(),
                // Sistem terakhir yang menerapkan hasil akhir Velocity ke posisi Transform.
                movement_system,
            ),
        )
        .run();
}

// --- COMPONENTS ---
// Komponen ini mendefinisikan data untuk entitas kita.

// Komponen umum untuk semua agen yang bisa bergerak
#[derive(Component)]
struct Agent {
    max_speed: f32,
    max_force: f32,
}

// Kecepatan saat ini dari sebuah entitas
#[derive(Component, Default, Deref, DerefMut)]
struct Velocity(Vec3);

// Komponen penanda untuk pemain
#[derive(Component)]
struct Player;

// --- BEHAVIOR COMPONENTS ---
// Komponen ini bertindak sebagai "tag" untuk memberitahu sistem
// perilaku mana yang harus diterapkan pada NPC.

#[derive(Component)]
struct Seek {
    target: Entity,
}

#[derive(Component)]
struct Flee {
    target: Entity,
}

#[derive(Component)]
struct Arrive {
    target: Entity,
    slowing_radius: f32,
}

#[derive(Component)]
struct Wander {
    circle_distance: f32,
    circle_radius: f32,
    wander_angle: f32,
    angle_change: f32,
}

#[derive(Component)]
struct Pursuit {
    target: Entity,
}

#[derive(Component)]
struct Evade {
    target: Entity,
}

// --- SETUP SYSTEM ---
// Fungsi ini hanya berjalan sekali saat aplikasi dimulai.
// Tugasnya adalah membuat semua objek awal di dalam scene.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Spawn Player (Target utama)
    let player_entity = commands
        .spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::Capsule {
                    radius: 0.4,
                    depth: 1.0,
                    ..default()
                })),
                material: materials.add(Color::rgb(0.2, 0.5, 0.9).into()),
                transform: Transform::from_xyz(0.0, 1.0, 0.0),
                ..default()
            },
            Player,
            Velocity::default(),
        ))
        .id();

    // --- Spawn NPCs dengan Perilaku Berbeda ---

    // 1. SEEK (Merah) - Akan selalu bergerak lurus ke arah pemain.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::RED.into()),
            transform: Transform::from_xyz(-10.0, 0.5, -10.0),
            ..default()
        },
        Agent {
            max_speed: 3.5,
            max_force: 0.8,
        },
        Velocity::default(),
        Seek {
            target: player_entity,
        },
    ));

    // 2. FLEE (Kuning) - Akan selalu lari menjauh dari pemain.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::YELLOW.into()),
            transform: Transform::from_xyz(5.0, 0.5, 5.0),
            ..default()
        },
        Agent {
            max_speed: 3.0,
            max_force: 1.0,
        },
        Velocity::default(),
        Flee {
            target: player_entity,
        },
    ));

    // 3. ARRIVE (Hijau) - Akan menuju pemain dan melambat saat mendekat.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::GREEN.into()),
            transform: Transform::from_xyz(10.0, 0.5, -10.0),
            ..default()
        },
        Agent {
            max_speed: 4.0,
            max_force: 0.7,
        },
        Velocity::default(),
        Arrive {
            target: player_entity,
            slowing_radius: 5.0,
        },
    ));

    // 4. WANDER (Ungu) - Akan bergerak tanpa tujuan secara acak.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::PURPLE.into()),
            transform: Transform::from_xyz(-10.0, 0.5, 10.0),
            ..default()
        },
        Agent {
            max_speed: 1.5,
            max_force: 0.3,
        },
        Velocity::default(),
        Wander {
            circle_distance: 3.0,
            circle_radius: 1.5,
            wander_angle: 0.0,
            angle_change: 0.4,
        },
    ));

    // 5. PURSUIT (Oranye) - Akan memprediksi posisi pemain dan mengejarnya.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::ORANGE.into()),
            transform: Transform::from_xyz(15.0, 0.5, 15.0),
            ..default()
        },
        Agent {
            max_speed: 4.2,
            max_force: 0.9,
        },
        Velocity::default(),
        Pursuit {
            target: player_entity,
        },
    ));

    // 6. EVADE (Cyan) - Akan memprediksi posisi pemain dan menghindarinya.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::CYAN.into()),
            transform: Transform::from_xyz(0.0, 0.5, 10.0),
            ..default()
        },
        Agent {
            max_speed: 3.8,
            max_force: 1.1,
        },
        Velocity::default(),
        Evade {
            target: player_entity,
        },
    ));

    // Lantai
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(25.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });

    // Cahaya
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Kamera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-20.0, 25.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}

// --- BEHAVIOR SYSTEMS ---
// Setiap fungsi ini mengimplementasikan satu logika steering behavior.

// 1. SEEK SYSTEM
fn seek_system(
    mut agent_query: Query<(&mut Velocity, &Transform, &Agent, &Seek)>,
    target_query: Query<&Transform>,
) {
    for (mut velocity, transform, agent, seek) in agent_query.iter_mut() {
        if let Ok(target_transform) = target_query.get(seek.target) {
            let desired = target_transform.translation - transform.translation;
            let desired_velocity = desired.normalize_or_zero() * agent.max_speed;
            let steering = (desired_velocity - velocity.0).clamp_length_max(agent.max_force);
            velocity.0 += steering;
        }
    }
}

// 2. FLEE SYSTEM
fn flee_system(
    mut agent_query: Query<(&mut Velocity, &Transform, &Agent, &Flee)>,
    target_query: Query<&Transform>,
) {
    for (mut velocity, transform, agent, flee) in agent_query.iter_mut() {
        if let Ok(target_transform) = target_query.get(flee.target) {
            let desired = transform.translation - target_transform.translation;
            let desired_velocity = desired.normalize_or_zero() * agent.max_speed;
            let steering = (desired_velocity - velocity.0).clamp_length_max(agent.max_force);
            velocity.0 += steering;
        }
    }
}

// 3. ARRIVE SYSTEM
fn arrive_system(
    mut agent_query: Query<(&mut Velocity, &Transform, &Agent, &Arrive)>,
    target_query: Query<&Transform>,
) {
    for (mut velocity, transform, agent, arrive) in agent_query.iter_mut() {
        if let Ok(target_transform) = target_query.get(arrive.target) {
            let desired = target_transform.translation - transform.translation;
            let distance = desired.length();
            let desired_velocity = if distance < arrive.slowing_radius {
                desired.normalize_or_zero() * agent.max_speed * (distance / arrive.slowing_radius)
            } else {
                desired.normalize_or_zero() * agent.max_speed
            };
            let steering = (desired_velocity - velocity.0).clamp_length_max(agent.max_force);
            velocity.0 += steering;
        }
    }
}

// 4. WANDER SYSTEM
fn wander_system(mut query: Query<(&mut Velocity, &Transform, &Agent, &mut Wander)>) {
    let mut rng = rand::thread_rng();
    for (mut velocity, _transform, agent, mut wander) in query.iter_mut() {
        let circle_center = velocity.normalize_or_zero() * wander.circle_distance;

        let displacement = Vec3::new(wander.wander_angle.cos(), 0.0, wander.wander_angle.sin())
            * wander.circle_radius;

        wander.wander_angle += rng.gen_range(-wander.angle_change..wander.angle_change);

        let wander_force = (circle_center + displacement).normalize_or_zero() * agent.max_force;
        velocity.0 += wander_force;
    }
}

// 5. PURSUIT SYSTEM
fn pursuit_system(
    mut agent_query: Query<(&mut Velocity, &Transform, &Agent, &Pursuit), Without<Player>>,
    target_query: Query<(&Transform, &Velocity), With<Player>>,
) {
    for (mut velocity, transform, agent, pursuit) in agent_query.iter_mut() {
        if let Ok((target_transform, target_velocity)) = target_query.get(pursuit.target) {
            let distance = (target_transform.translation - transform.translation).length();
            let prediction_time = distance / agent.max_speed;
            let future_position =
                target_transform.translation + target_velocity.0 * prediction_time;

            let desired = future_position - transform.translation;
            let desired_velocity = desired.normalize_or_zero() * agent.max_speed;
            let steering = (desired_velocity - velocity.0).clamp_length_max(agent.max_force);
            velocity.0 += steering;
        }
    }
}

// 6. EVADE SYSTEM
fn evade_system(
    mut agent_query: Query<(&mut Velocity, &Transform, &Agent, &Evade), Without<Player>>,
    target_query: Query<(&Transform, &Velocity), With<Player>>,
) {
    for (mut velocity, transform, agent, evade) in agent_query.iter_mut() {
        if let Ok((target_transform, target_velocity)) = target_query.get(evade.target) {
            let distance = (target_transform.translation - transform.translation).length();
            let prediction_time = distance / agent.max_speed;
            let future_position =
                target_transform.translation + target_velocity.0 * prediction_time;

            let desired = transform.translation - future_position;
            let desired_velocity = desired.normalize_or_zero() * agent.max_speed;
            let steering = (desired_velocity - velocity.0).clamp_length_max(agent.max_force);
            velocity.0 += steering;
        }
    }
}

// --- COMBINATION SYSTEMS ---

// SEPARATION SYSTEM
// Mencegah NPC saling menabrak.
fn separation_system(mut query: Query<(Entity, &mut Velocity, &Transform, &Agent)>) {
    let mut combinations = query.iter_combinations_mut();
    while let Some([(_, mut v1, t1, a1), (_, mut v2, t2, a2)]) = combinations.fetch_next() {
        let distance = t1.translation.distance(t2.translation);

        if distance > 0.0 && distance < DESIRED_SEPARATION {
            // Hitung gaya tolak yang berbanding terbalik dengan jarak
            let separation_force = (t1.translation - t2.translation).normalize_or_zero() / distance;

            // Terapkan gaya ke kedua agen
            v1.0 += separation_force * a1.max_force;
            v2.0 -= separation_force * a2.max_force; // Gaya berlawanan
        }
    }
}

// CONTAINMENT SYSTEM
// Mencegah agen keluar dari batas peta.
fn containment_system(mut query: Query<(&mut Velocity, &Transform, &Agent)>) {
    const MAP_BOUNDARY: f32 = 12.0; // Setengah dari ukuran peta (25.0 / 2) dikurangi sedikit

    for (mut velocity, transform, agent) in query.iter_mut() {
        let mut desired_change = Vec3::ZERO;

        // Cek batas X
        if transform.translation.x > MAP_BOUNDARY {
            desired_change.x = -agent.max_speed;
        } else if transform.translation.x < -MAP_BOUNDARY {
            desired_change.x = agent.max_speed;
        }

        // Cek batas Z
        if transform.translation.z > MAP_BOUNDARY {
            desired_change.z = -agent.max_speed;
        } else if transform.translation.z < -MAP_BOUNDARY {
            desired_change.z = agent.max_speed;
        }

        if desired_change != Vec3::ZERO {
            let steer = (desired_change - velocity.0).clamp_length_max(agent.max_force * 2.0); // Beri gaya lebih kuat
            velocity.0 += steer;
        }
    }
}

// --- UTILITY SYSTEMS ---

// MOVEMENT SYSTEM
// Sistem ini menerapkan Velocity akhir ke Transform (posisi) dan
// memutar agen agar menghadap ke arah gerakannya.
fn movement_system(mut query: Query<(&mut Transform, &mut Velocity, &Agent)>, time: Res<Time>) {
    for (mut transform, mut velocity, agent) in query.iter_mut() {
        // Batasi kecepatan maksimum
        velocity.0 = velocity.0.clamp_length_max(agent.max_speed);

        // Perbarui posisi
        transform.translation += velocity.0 * time.delta_seconds();

        // Kunci posisi Y agar tidak menembus lantai
        transform.translation.y = 0.5;

        // Putar entitas untuk menghadap ke arah gerakan
        if velocity.0.length_squared() > 0.0 {
            let look_at = velocity.0.normalize();
            transform.look_to(look_at, Vec3::Y);
        }
    }
}

// PLAYER MOVEMENT SYSTEM
// Mengizinkan Anda mengontrol pemain dengan tombol panah/WASD.
fn player_movement_system(
    keyboard_input: Res<Input<KeyCode>>,
    mut query: Query<&mut Transform, With<Player>>,
    time: Res<Time>,
) {
    if let Ok(mut transform) = query.get_single_mut() {
        let mut direction = Vec3::ZERO;
        if keyboard_input.pressed(KeyCode::Up) || keyboard_input.pressed(KeyCode::W) {
            direction.z -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::Down) || keyboard_input.pressed(KeyCode::S) {
            direction.z += 1.0;
        }
        if keyboard_input.pressed(KeyCode::Left) || keyboard_input.pressed(KeyCode::A) {
            direction.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::Right) || keyboard_input.pressed(KeyCode::D) {
            direction.x += 1.0;
        }

        let movement = direction.normalize_or_zero() * PLAYER_SPEED * time.delta_seconds();
        transform.translation += movement;

        transform.translation.y = 1.0;

        if direction.length_squared() > 0.0 {
            transform.look_to(direction.normalize(), Vec3::Y);
        }
    }
}
