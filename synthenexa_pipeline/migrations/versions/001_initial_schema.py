"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create experiments table
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(length=50), nullable=False),
        sa.Column('experiment_name', sa.String(length=255), nullable=True),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('base_prompt_id', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('config', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id')
    )
    op.create_index('idx_experiments_run_id', 'experiments', ['run_id'], unique=False)
    op.create_index('idx_experiments_created_at', 'experiments', ['created_at'], unique=False)

    # Create prompts table
    op.create_table(
        'prompts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prompt_id', sa.String(length=50), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('is_base', sa.Boolean(), nullable=True),
        sa.Column('is_best', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prompt_id', 'experiment_id', 'generation', name='uq_prompt_experiment_generation')
    )
    op.create_index('idx_prompts_experiment_generation', 'prompts', ['experiment_id', 'generation'], unique=False)
    op.create_index('idx_prompts_is_best', 'prompts', ['is_best'], unique=False, postgresql_where=sa.text("is_best = TRUE"))

    # Create evaluations table
    op.create_table(
        'evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prompt_id', sa.String(length=50), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('passed', sa.Integer(), nullable=False),
        sa.Column('failed', sa.Integer(), nullable=False),
        sa.Column('total', sa.Integer(), nullable=False),
        sa.Column('syntax_errors', sa.Integer(), nullable=True),
        sa.Column('errors', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_evaluations_prompt_generation', 'evaluations', ['prompt_id', 'generation'], unique=False)
    op.create_index('idx_evaluations_experiment', 'evaluations', ['experiment_id'], unique=False)

    # Create snapshots table
    op.create_table(
        'snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('generation', sa.Integer(), nullable=False),
        sa.Column('best_score', sa.Float(), nullable=True),
        sa.Column('best_prompt_id', sa.String(length=50), nullable=True),
        sa.Column('best_prompt_text', sa.Text(), nullable=True),
        sa.Column('scores', postgresql.JSONB(), nullable=True),
        sa.Column('reward_history', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('experiment_id', 'generation', name='uq_snapshot_experiment_generation')
    )
    op.create_index('idx_snapshots_experiment', 'snapshots', ['experiment_id'], unique=False)

    # Create run_logs table
    op.create_table(
        'run_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('log_entry', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_run_logs_experiment', 'run_logs', ['experiment_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_run_logs_experiment', table_name='run_logs')
    op.drop_table('run_logs')
    
    op.drop_index('idx_snapshots_experiment', table_name='snapshots')
    op.drop_table('snapshots')
    
    op.drop_index('idx_evaluations_experiment', table_name='evaluations')
    op.drop_index('idx_evaluations_prompt_generation', table_name='evaluations')
    op.drop_table('evaluations')
    
    op.drop_index('idx_prompts_is_best', table_name='prompts')
    op.drop_index('idx_prompts_experiment_generation', table_name='prompts')
    op.drop_table('prompts')
    
    op.drop_index('idx_experiments_created_at', table_name='experiments')
    op.drop_index('idx_experiments_run_id', table_name='experiments')
    op.drop_table('experiments')
