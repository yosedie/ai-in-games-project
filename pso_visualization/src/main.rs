use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::PresentMode;
use rand::Rng;

const DOMAIN: f32 = 30.0;
const PARTICLE_SIZE: f32 = 0.7;
const TARGET_SIZE: f32 = 1.5;
const LERP_SPEED: f32 = 4.5; // Kecepatan smooth movement (1.0-10.0)

#[derive(Clone, Copy)]
struct PsoParams {
    population: usize,
    generations: usize,
    w: f32,
    c1: f32,
    c2: f32,
}

impl Default for PsoParams {
    fn default() -> Self {
        Self {
            population: 10,
            generations: 15,
            w: 0.6,
            c1: 1.8,
            c2: 2.1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Particle {
    position: Vec2,        // Current visual position (smooth)
    target_position: Vec2, // Target position after PSO calculation
    velocity: Vec2,
    pbest_pos: Vec2,
    pbest_val: f32,
}

#[derive(Resource)]
struct PsoState {
    params: PsoParams,
    particles: Vec<Particle>,
    gbest_pos: Vec2,
    gbest_val: f32,
    current_gen: usize,
    paused: bool,
    converged: bool,
    target: Option<Vec2>,
}

#[derive(Component)]
struct ParticleMarker(usize);
#[derive(Component)]
struct TargetMarker;
#[derive(Component)]
struct GenText;
#[derive(Component)]
struct ControlsText;
#[derive(Component)]
struct FpsText;

#[derive(Resource, Default)]
struct ClickMarker(pub Option<Vec2>);

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.025, 0.028, 0.058)))
        .insert_resource(PsoState {
            params: PsoParams::default(),
            particles: vec![],
            gbest_pos: Vec2::ZERO,
            gbest_val: f32::INFINITY,
            current_gen: 0,
            paused: true,
            converged: false,
            target: None,
        })
        .insert_resource(ClickMarker(None))
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "PSO Visualization - Smooth Animation".to_string(),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera_controls,
                mouse_set_target,
                update_generation_text,
                update_fps_text,
                update_ui_sliders,
                update_particles_visual,
                pso_tick,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 38.0, 38.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::WHITE,
            illuminance: 15000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.2, 1.7, 0.0)),
        ..default()
    });

    // Title
    commands.spawn((
        TextBundle::from_section(
            "Particle Swarm Optimization (Smooth)\nKennedy & Eberhart (1995)",
            TextStyle {
                font_size: 23.0,
                color: Color::YELLOW,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(9.0),
            left: Val::Px(18.0),
            ..default()
        }),
        ControlsText,
    ));

    // Controls
    commands.spawn((
        TextBundle::from_section(
            "Controls:
Click = Set Target
[G] step/auto   [P] pause
[+][-] generations
[U][J] pop ±   [I][K] w ±
[O][L] c1 ±   [;][P] c2 ±
[N] new random
[ESC] exit",
            TextStyle {
                font_size: 14.0,
                color: Color::rgb(0.85, 0.9, 1.0),
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(74.0),
            left: Val::Px(18.0),
            ..default()
        }),
        ControlsText,
    ));

    // Gen info
    commands.spawn((
        TextBundle::from_section(
            "",
            TextStyle {
                font_size: 18.0,
                color: Color::GOLD,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(18.0),
            left: Val::Px(18.0),
            ..default()
        }),
        GenText,
    ));

    // FPS counter
    commands.spawn((
        TextBundle::from_section(
            "FPS: --",
            TextStyle {
                font_size: 16.0,
                color: Color::LIME_GREEN,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(9.0),
            right: Val::Px(18.0),
            ..default()
        }),
        FpsText,
    ));
}

fn camera_controls(
    mut query: Query<&mut Transform, With<Camera3d>>,
    keyboard: Res<Input<KeyCode>>,
    time: Res<Time>,
) {
    let speed = 24.0 * time.delta_seconds();
    for mut trans in query.iter_mut() {
        let mut move_dir = Vec3::ZERO;
        if keyboard.pressed(KeyCode::A) {
            move_dir.x -= 1.0;
        }
        if keyboard.pressed(KeyCode::D) {
            move_dir.x += 1.0;
        }
        if keyboard.pressed(KeyCode::W) {
            move_dir.z -= 1.0;
        }
        if keyboard.pressed(KeyCode::S) {
            move_dir.z += 1.0;
        }
        if keyboard.pressed(KeyCode::Q) {
            move_dir.y -= 1.0;
        }
        if keyboard.pressed(KeyCode::E) {
            move_dir.y += 1.0;
        }
        trans.translation += move_dir * speed;
    }
}

fn mouse_set_target(
    mut click_marker: ResMut<ClickMarker>,
    windows: Query<&Window>,
    mouse: Res<Input<MouseButton>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mut commands: Commands,
    mut pso: ResMut<PsoState>,
    particles_query: Query<Entity, With<ParticleMarker>>,
    target_entity: Query<Entity, With<TargetMarker>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let window = windows.single();
    if mouse.just_pressed(MouseButton::Left) {
        if let Some(cursor) = window.cursor_position() {
            let (camera, camera_transform) = camera_query.single();
            if let Some(ray) = camera.viewport_to_world(camera_transform, cursor) {
                let t = -ray.origin.y / ray.direction.y;
                let pos = ray.origin + ray.direction * t;
                let pos2d = Vec2::new(pos.x, pos.z);
                click_marker.0 = Some(pos2d);

                // Target marker
                let mark_color = Color::rgb(1.0, 0.15, 0.15);
                if let Ok(e) = target_entity.get_single() {
                    commands
                        .entity(e)
                        .insert(Transform::from_xyz(pos2d.x, 1.1, pos2d.y));
                } else {
                    commands.spawn((
                        PbrBundle {
                            mesh: meshes.add(Mesh::from(shape::UVSphere {
                                radius: TARGET_SIZE,
                                sectors: 20,
                                stacks: 20,
                            })),
                            material: materials.add(StandardMaterial {
                                base_color: mark_color,
                                emissive: mark_color,
                                ..default()
                            }),
                            transform: Transform::from_xyz(pos2d.x, 1.1, pos2d.y),
                            ..default()
                        },
                        TargetMarker,
                    ));
                }

                // Despawn old particles
                for e in particles_query.iter() {
                    commands.entity(e).despawn_recursive();
                }

                pso.target = Some(pos2d);
                pso.paused = true;
                pso.converged = false;
                pso.current_gen = 0;
                pso.gbest_val = f32::INFINITY;
                pso.particles = init_population(&pso.params);
                render_particles(&mut commands, &mut meshes, &mut materials, &pso.particles);
            }
        }
    }
}

fn init_population(params: &PsoParams) -> Vec<Particle> {
    let mut rng = rand::thread_rng();
    (0..params.population)
        .map(|_| {
            let pos = Vec2::new(
                rng.gen_range(-DOMAIN..DOMAIN),
                rng.gen_range(-DOMAIN..DOMAIN),
            );
            Particle {
                position: pos,
                target_position: pos,
                velocity: Vec2::ZERO,
                pbest_pos: pos,
                pbest_val: f32::INFINITY,
            }
        })
        .collect()
}

fn render_particles(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    particles: &[Particle],
) {
    for (i, part) in particles.iter().enumerate() {
        let hue = i as f32 / particles.len() as f32;
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(shape::UVSphere {
                    radius: PARTICLE_SIZE,
                    sectors: 14,
                    stacks: 14,
                })),
                material: materials.add(StandardMaterial {
                    base_color: Color::hsl(200.0 + hue * 120.0, 0.8, 0.65),
                    emissive: Color::rgb(0.1, 0.2, 0.5),
                    ..default()
                }),
                transform: Transform::from_xyz(part.position.x, 1.0, part.position.y),
                ..default()
            },
            ParticleMarker(i),
        ));
    }
}

// SMOOTH INTERPOLATION HERE!
fn update_particles_visual(
    mut particles_query: Query<(&ParticleMarker, &mut Transform)>,
    mut pso: ResMut<PsoState>,
    time: Res<Time>,
) {
    for (marker, mut transform) in particles_query.iter_mut() {
        if let Some(part) = pso.particles.get_mut(marker.0) {
            // Lerp dari position ke target_position
            part.position = part
                .position
                .lerp(part.target_position, LERP_SPEED * time.delta_seconds());

            transform.translation.x = part.position.x;
            transform.translation.z = part.position.y;
        }
    }
}

fn update_generation_text(mut text_query: Query<&mut Text, With<GenText>>, pso: Res<PsoState>) {
    let mut text = text_query.single_mut();
    let params = &pso.params;
    text.sections[0].value = format!(
        "Gen: {}/{}  |  Pop: {}  |  w: {:.2}  c1: {:.2}  c2: {:.2}  {}",
        pso.current_gen,
        params.generations,
        params.population,
        params.w,
        params.c1,
        params.c2,
        if pso.converged { " ✅ CONVERGED!" } else { "" }
    );
}

fn update_fps_text(
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
) {
    for mut text in query.iter_mut() {
        if let Some(fps) = diagnostics.get(bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                text.sections[0].value = format!("FPS: {:.0}", value);
            }
        }
    }
}

fn pso_tick(time: Res<Time>, keyboard: Res<Input<KeyCode>>, mut pso: ResMut<PsoState>) {
    if pso.target.is_none() || pso.converged {
        return;
    }

    let mut advance = false;
    if keyboard.just_pressed(KeyCode::G) {
        advance = true;
        pso.paused = false;
    }
    if keyboard.just_pressed(KeyCode::P) {
        pso.paused = !pso.paused;
    }

    // Update tiap 0.3 detik untuk smooth animation
    if !pso.paused && (time.elapsed_seconds_f64() % 0.3 < 0.02) {
        advance = true;
    }

    if !advance {
        return;
    }

    // Copy params untuk avoid borrow issue
    let params = pso.params;
    let goal = pso.target.unwrap();

    // 1. Update pbest & gbest
    let mut global_best_val = f32::INFINITY;
    let mut global_best_pos = Vec2::ZERO;

    for part in &mut pso.particles {
        // Use target_position untuk fitness (posisi sebenarnya dalam algoritma)
        let dist = (part.target_position - goal).length();
        if dist < part.pbest_val {
            part.pbest_pos = part.target_position;
            part.pbest_val = dist;
        }
        if dist < global_best_val {
            global_best_val = dist;
            global_best_pos = part.target_position;
        }
    }

    pso.gbest_val = global_best_val;
    pso.gbest_pos = global_best_pos;

    // 2. Update velocity & target_position
    let mut rng = rand::thread_rng();
    for part in &mut pso.particles {
        let r1 = rng.gen_range(0.0..1.0);
        let r2 = rng.gen_range(0.0..1.0);

        part.velocity = params.w * part.velocity
            + params.c1 * r1 * (part.pbest_pos - part.target_position)
            + params.c2 * r2 * (global_best_pos - part.target_position);

        let mut new_pos = part.target_position + part.velocity;
        new_pos.x = new_pos.x.clamp(-DOMAIN, DOMAIN);
        new_pos.y = new_pos.y.clamp(-DOMAIN, DOMAIN);

        part.target_position = new_pos; // Set target untuk lerp
    }

    pso.current_gen += 1;

    if pso.current_gen >= params.generations || pso.gbest_val < 0.7 {
        pso.converged = true;
        pso.paused = true;
    }
}

fn update_ui_sliders(
    keyboard: Res<Input<KeyCode>>,
    mut pso: ResMut<PsoState>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    particles_query: Query<Entity, With<ParticleMarker>>,
) {
    if keyboard.just_pressed(KeyCode::Equals) {
        pso.params.generations += 2;
    }
    if keyboard.just_pressed(KeyCode::Minus) {
        pso.params.generations = pso.params.generations.saturating_sub(2);
    }
    if keyboard.just_pressed(KeyCode::U) {
        pso.params.population += 1;
    }
    if keyboard.just_pressed(KeyCode::J) {
        pso.params.population = pso.params.population.saturating_sub(1).max(3);
    }
    if keyboard.just_pressed(KeyCode::I) {
        pso.params.w = (pso.params.w + 0.05).min(1.2);
    }
    if keyboard.just_pressed(KeyCode::K) {
        pso.params.w = (pso.params.w - 0.05).max(0.0);
    }
    if keyboard.just_pressed(KeyCode::O) {
        pso.params.c1 += 0.1;
    }
    if keyboard.just_pressed(KeyCode::L) {
        pso.params.c1 = (pso.params.c1 - 0.1).max(0.0);
    }
    if keyboard.just_pressed(KeyCode::P) {
        pso.params.c2 += 0.1;
    }
    if keyboard.just_pressed(KeyCode::Semicolon) {
        pso.params.c2 = (pso.params.c2 - 0.1).max(0.0);
    }

    if keyboard.just_pressed(KeyCode::N) {
        pso.paused = true;
        pso.converged = false;
        pso.current_gen = 0;
        pso.gbest_val = f32::INFINITY;
        if pso.target.is_some() {
            for e in particles_query.iter() {
                commands.entity(e).despawn_recursive();
            }
            pso.particles = init_population(&pso.params);
            render_particles(&mut commands, &mut meshes, &mut materials, &pso.particles);
        }
    }
}
